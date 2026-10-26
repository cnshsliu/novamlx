import Foundation

/// Full-string regular expression used by Flint tool schemas (`pattern` / `format`).
///
/// JSON Schema patterns are matched against the entire argument, the same way
/// xgrammar anchors a `pattern`. `isPrefix` is true when some continuation of
/// the text can still match, so a filtered packet can mask illegal tokens
/// before the value is closed.
struct FlintPattern: Equatable, Sendable {
    let source: String
    private let start: Int
    private let accept: Int
    private let edges: [Edge]

    struct Edge: Equatable, Sendable {
        enum Kind: Equatable, Sendable {
            case epsilon
            case begin
            case end
            case char(Character)
            case any
            case cls(negated: Bool, items: [ClassItem])
        }

        var kind: Kind
        var dest: Int
    }

    struct ClassItem: Equatable, Sendable {
        var lower: Character
        var upper: Character
    }

    init?(_ source: String) {
        var builder = Builder()
        guard let node = builder.parse(source) else { return nil }
        let compiler = Compiler()
        guard compiler.compile(node, from: compiler.start, to: compiler.accept) else { return nil }
        self.source = source
        self.start = compiler.start
        self.accept = compiler.accept
        self.edges = compiler.edges
    }

    func matches(_ text: String) -> Bool {
        run(text).complete
    }

    func isPrefix(_ text: String) -> Bool {
        run(text).prefix
    }

    private func run(_ text: String) -> (prefix: Bool, complete: Bool) {
        let chars = Array(text)
        var states = closure([start], atBegin: true, atEnd: chars.isEmpty)
        if states.isEmpty { return (false, false) }
        for (index, char) in chars.enumerated() {
            var next: [Int] = []
            for state in states {
                for edge in edges[state] where takes(edge, char: char) {
                    next.append(edge.dest)
                }
            }
            let atEnd = index == chars.count - 1
            states = closure(next, atBegin: false, atEnd: atEnd)
            if states.isEmpty { return (false, false) }
        }
        let complete = states.contains(accept)
        let prefix = complete || reachesAccept(states)
        return (prefix, complete)
    }

    private func takes(_ edge: Edge, char: Character) -> Bool {
        switch edge.kind {
        case .char(let expected):
            return expected == char
        case .any:
            return true
        case .cls(let negated, let items):
            let hit = items.contains { char >= $0.lower && char <= $0.upper }
            return negated ? !hit : hit
        case .epsilon, .begin, .end:
            return false
        }
    }

    private func closure(_ seeds: [Int], atBegin: Bool, atEnd: Bool) -> [Int] {
        var seen = Set<Int>()
        var stack = seeds
        for seed in seeds { seen.insert(seed) }
        while let state = stack.popLast() {
            guard edges.indices.contains(state) else { continue }
            for edge in edges[state] {
                let take: Bool
                switch edge.kind {
                case .epsilon: take = true
                case .begin: take = atBegin
                case .end: take = atEnd
                default: take = false
                }
                if take, seen.insert(edge.dest).inserted {
                    stack.append(edge.dest)
                }
            }
        }
        return Array(seen)
    }

    /// Some continuation, including ending the string now via `$`, can accept.
    private func reachesAccept(_ seeds: [Int]) -> Bool {
        var seen = Set<Int>()
        var stack = seeds
        for seed in seeds { seen.insert(seed) }
        while let state = stack.popLast() {
            if state == accept { return true }
            guard edges.indices.contains(state) else { continue }
            for edge in edges[state] where seen.insert(edge.dest).inserted {
                stack.append(edge.dest)
            }
        }
        return false
    }
}

extension FlintPattern {
    fileprivate enum Node {
        case empty
        case char(Character)
        case any
        case begin
        case end
        case cls(negated: Bool, items: [ClassItem])
        case concat([Node])
        case alt([Node])
        case rep(Node, min: Int, max: Int?)
    }

    fileprivate struct Builder {
        var index = 0
        var chars: [Character] = []

        mutating func parse(_ source: String) -> Node? {
            Compiler().parseSource(source)
        }

    }

    fileprivate final class Compiler {
        var edges: [[Edge]] = [[], []]
        let start = 0
        let accept = 1

        func fresh() -> Int {
            edges.append([])
            return edges.count - 1
        }

        func link(_ from: Int, _ kind: Edge.Kind, _ to: Int) {
            edges[from].append(Edge(kind: kind, dest: to))
        }

        var index = 0
        var chars: [Character] = []

        func parseSource(_ source: String) -> Node? {
            chars = Array(source)
            index = 0
            guard let node = parseAlt() else { return nil }
            if index != chars.count { return nil }
            return node
        }

        func compile(_ node: Node, from start: Int, to accept: Int) -> Bool {
            switch node {
            case .empty:
                link(start, .epsilon, accept)
            case .char(let char):
                link(start, .char(char), accept)
            case .any:
                link(start, .any, accept)
            case .begin:
                link(start, .begin, accept)
            case .end:
                link(start, .end, accept)
            case .cls(let negated, let items):
                link(start, .cls(negated: negated, items: items), accept)
            case .concat(let parts):
                guard !parts.isEmpty else {
                    link(start, .epsilon, accept)
                    return true
                }
                var current = start
                for (offset, part) in parts.enumerated() {
                    let next = offset == parts.count - 1 ? accept : fresh()
                    if !compile(part, from: current, to: next) { return false }
                    current = next
                }
            case .alt(let parts):
                for part in parts {
                    let head = fresh()
                    let tail = fresh()
                    link(start, .epsilon, head)
                    if !compile(part, from: head, to: tail) { return false }
                    link(tail, .epsilon, accept)
                }
            case .rep(let child, let minimum, let maximum):
                var current = start
                for _ in 0..<minimum {
                    let next = fresh()
                    if !compile(child, from: current, to: next) { return false }
                    current = next
                }
                if maximum == nil {
                    let split = fresh()
                    let bodyEnd = fresh()
                    link(current, .epsilon, split)
                    link(split, .epsilon, accept)
                    if !compile(child, from: split, to: bodyEnd) { return false }
                    link(bodyEnd, .epsilon, split)
                } else if let maximum {
                    link(current, .epsilon, accept)
                    let extra = max(0, maximum - minimum)
                    for _ in 0..<extra {
                        let next = fresh()
                        if !compile(child, from: current, to: next) { return false }
                        link(next, .epsilon, accept)
                        current = next
                    }
                }
            }
            return true
        }

        private func peek() -> Character? {
            index < chars.count ? chars[index] : nil
        }

        private func get() -> Character? {
            guard index < chars.count else { return nil }
            let char = chars[index]
            index += 1
            return char
        }

        private func parseAlt() -> Node? {
            guard let first = parseConcat() else { return nil }
            var parts = [first]
            while peek() == "|" {
                _ = get()
                guard let next = parseConcat() else { return nil }
                parts.append(next)
            }
            if parts.count == 1 { return parts[0] }
            return .alt(parts)
        }

        private func parseConcat() -> Node? {
            var parts: [Node] = []
            while let char = peek(), char != "|" && char != ")" {
                guard let part = parseRepeat() else { return nil }
                parts.append(part)
            }
            if parts.isEmpty { return .empty }
            if parts.count == 1 { return parts[0] }
            return .concat(parts)
        }

        private func parseRepeat() -> Node? {
            guard let atom = parseAtom() else { return nil }
            switch peek() {
            case "*":
                _ = get()
                return .rep(atom, min: 0, max: nil)
            case "+":
                _ = get()
                return .rep(atom, min: 1, max: nil)
            case "?":
                _ = get()
                return .rep(atom, min: 0, max: 1)
            case "{":
                _ = get()
                let minimum = readNumber() ?? 0
                if peek() == "}" {
                    _ = get()
                    return .rep(atom, min: minimum, max: minimum)
                }
                guard peek() == "," else { return nil }
                _ = get()
                let maximum = readNumber()
                guard peek() == "}" else { return nil }
                _ = get()
                return .rep(atom, min: minimum, max: maximum)
            default:
                return atom
            }
        }

        private func readNumber() -> Int? {
            var digits = ""
            while let char = peek(), char >= "0" && char <= "9" {
                digits.append(get()!)
            }
            return digits.isEmpty ? nil : Int(digits)
        }

        private func parseAtom() -> Node? {
            guard let char = peek() else { return nil }
            if char == "(" {
                _ = get()
                guard let node = parseAlt() else { return nil }
                guard peek() == ")" else { return nil }
                _ = get()
                return node
            }
            if char == "^" {
                _ = get()
                return .begin
            }
            if char == "$" {
                _ = get()
                return .end
            }
            if char == "." {
                _ = get()
                return .any
            }
            if char == "[" {
                return parseClass()
            }
            if char == "\\" {
                _ = get()
                guard let escaped = get() else { return nil }
                return escape(escaped)
            }
            _ = get()
            return .char(char)
        }

        private func parseClass() -> Node? {
            guard get() == "[" else { return nil }
            var negated = false
            if peek() == "^" {
                negated = true
                _ = get()
            }
            var items: [ClassItem] = []
            while let char = peek(), char != "]" {
                if char == "\\" {
                    _ = get()
                    guard let escaped = get() else { return nil }
                    items.append(ClassItem(lower: escaped, upper: escaped))
                    continue
                }
                let lower = get()!
                if peek() == "-", index + 1 < chars.count, chars[index + 1] != "]" {
                    _ = get()
                    guard let upper = get() else { return nil }
                    items.append(ClassItem(lower: lower, upper: upper))
                } else {
                    items.append(ClassItem(lower: lower, upper: lower))
                }
            }
            guard get() == "]" else { return nil }
            return .cls(negated: negated, items: items)
        }

        private func escape(_ char: Character) -> Node {
            switch char {
            case "d":
                return .cls(negated: false, items: [ClassItem(lower: "0", upper: "9")])
            case "D":
                return .cls(negated: true, items: [ClassItem(lower: "0", upper: "9")])
            case "w":
                return .cls(negated: false, items: [
                    ClassItem(lower: "A", upper: "Z"),
                    ClassItem(lower: "a", upper: "z"),
                    ClassItem(lower: "0", upper: "9"),
                    ClassItem(lower: "_", upper: "_"),
                ])
            case "W":
                return .cls(negated: true, items: [
                    ClassItem(lower: "A", upper: "Z"),
                    ClassItem(lower: "a", upper: "z"),
                    ClassItem(lower: "0", upper: "9"),
                    ClassItem(lower: "_", upper: "_"),
                ])
            case "s":
                return .cls(negated: false, items: [
                    ClassItem(lower: " ", upper: " "),
                    ClassItem(lower: "\t", upper: "\t"),
                    ClassItem(lower: "\n", upper: "\n"),
                    ClassItem(lower: "\r", upper: "\r"),
                ])
            case "S":
                return .cls(negated: true, items: [
                    ClassItem(lower: " ", upper: " "),
                    ClassItem(lower: "\t", upper: "\t"),
                    ClassItem(lower: "\n", upper: "\n"),
                    ClassItem(lower: "\r", upper: "\r"),
                ])
            default:
                return .char(char)
            }
        }
    }
}

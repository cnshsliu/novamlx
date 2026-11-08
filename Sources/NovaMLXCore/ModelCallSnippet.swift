import Foundation

/// Plain-text curl, Python, and TypeScript calls for one loaded model.
public enum ModelCallSnippet {
    public static func text(
        modelId: String,
        modelType: ModelType?,
        family: ModelFamily?,
        baseURL: String,
        apiKey: String
    ) -> String {
        let kind = kind(modelType: modelType, family: family)
        let root = baseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let url = "\(root)\(kind.path)"
        let key = apiKey.isEmpty ? "YOUR_API_KEY" : apiKey
        let quoted = jsonString(modelId)
        return """
        NovaMLX
        Model: \(modelId)
        POST \(url)

        --- curl ---
        \(curl(kind: kind, url: url, key: key, model: quoted))

        --- Python ---
        \(python(kind: kind, url: url, key: key, model: quoted))

        --- TypeScript ---
        \(typescript(kind: kind, url: url, key: key, model: quoted))
        """
    }

    private enum Kind {
        case chat
        case vision
        case embeddings
        case speech
        case transcription
        case image
        case matting
        case threed
        case decision

        var path: String {
            switch self {
            case .chat, .vision: return "/v1/chat/completions"
            case .embeddings: return "/v1/embeddings"
            case .speech: return "/v1/audio/speech"
            case .transcription: return "/v1/audio/transcriptions"
            case .image: return "/v1/images/generations"
            case .matting: return "/v1/images/matting"
            case .threed: return "/v1/3d/generations"
            case .decision: return "/v1/decisions"
            }
        }
    }

    private static func kind(modelType: ModelType?, family: ModelFamily?) -> Kind {
        switch modelType {
        case .vlm: return .vision
        case .embedding: return .embeddings
        case .decision: return .decision
        case .threed: return .threed
        case .image: return family == .biRefNet ? .matting : .image
        case .audio:
            switch family {
            case .whisper, .qwen3Asr: return .transcription
            default: return .speech
            }
        default: return .chat
        }
    }

    /// JSON string literal, including the surrounding quotes.
    private static func jsonString(_ value: String) -> String {
        var out = "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\\": out += "\\\\"
            case "\"": out += "\\\""
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            default: out.unicodeScalars.append(scalar)
            }
        }
        out += "\""
        return out
    }

    private static func shellSingleQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    private static func body(kind: Kind, model: String) -> String {
        switch kind {
        case .chat:
            return "{\"model\":\(model),\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":256}"
        case .vision:
            return "{\"model\":\(model),\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is in this image?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://example.com/photo.jpg\"}}]}],\"max_tokens\":256}"
        case .embeddings:
            return "{\"model\":\(model),\"input\":\"Hello\"}"
        case .image:
            return "{\"model\":\(model),\"prompt\":\"A red apple on a wooden table\",\"size\":\"1024x1024\"}"
        case .matting:
            return "{\"model\":\(model),\"image\":\"BASE64_PNG\"}"
        case .threed:
            return "{\"model\":\(model),\"image\":\"/path/to/photo.png\"}"
        case .decision:
            return "{\"model\":\(model),\"state\":\"The customer wants a refund.\",\"questions\":{\"department\":{\"type\":\"choice\",\"instructions\":\"Which department should handle this?\",\"criteria\":[\"billing\",\"technical\"]}}}"
        case .speech, .transcription:
            return ""
        }
    }

    private static func curl(kind: Kind, url: String, key: String, model: String) -> String {
        let auth = "-H \"Authorization: Bearer \(key)\""
        switch kind {
        case .transcription:
            return """
            curl -s -X POST \(url) \\
              \(auth) \\
              -F file=@recording.wav \\
              -F model=\(model)
            """
        case .speech:
            let payload = "{\"model\":\(model),\"input\":\"Hello\",\"voice\":\"Tingting\"}"
            return """
            curl -s -X POST \(url) \\
              \(auth) \\
              -H "Content-Type: application/json" \\
              -d \(shellSingleQuoted(payload)) \\
              --output speech.wav
            """
        default:
            return """
            curl -s -X POST \(url) \\
              \(auth) \\
              -H "Content-Type: application/json" \\
              -d \(shellSingleQuoted(body(kind: kind, model: model)))
            """
        }
    }

    private static func python(kind: Kind, url: String, key: String, model: String) -> String {
        switch kind {
        case .transcription:
            return """
            import requests

            resp = requests.post(
                "\(url)",
                headers={"Authorization": "Bearer \(key)"},
                files={"file": open("recording.wav", "rb")},
                data={"model": \(model)},
            )
            print(resp.json())
            """
        case .speech:
            return """
            import requests

            resp = requests.post(
                "\(url)",
                headers={"Authorization": "Bearer \(key)"},
                json={"model": \(model), "input": "Hello", "voice": "Tingting"},
            )
            open("speech.wav", "wb").write(resp.content)
            """
        case .threed:
            return """
            import requests

            headers = {"Authorization": "Bearer \(key)"}
            resp = requests.post(
                "\(url)",
                headers=headers,
                json={"model": \(model), "image": "/path/to/photo.png"},
            )
            job = resp.json()
            print(job)
            print(requests.get(f"\(url)/{job['id']}", headers=headers).json())
            """
        default:
            let payload = body(kind: kind, model: model).replacingOccurrences(of: "\\", with: "\\\\")
            return """
            import json
            import requests

            resp = requests.post(
                "\(url)",
                headers={"Authorization": "Bearer \(key)"},
                json=json.loads(\"\"\"\(payload)\"\"\"),
            )
            print(resp.json())
            """
        }
    }

    private static func typescript(kind: Kind, url: String, key: String, model: String) -> String {
        switch kind {
        case .transcription:
            return """
            import { readFile } from "node:fs/promises";

            const form = new FormData();
            form.append("file", new Blob([await readFile("recording.wav")]), "recording.wav");
            form.append("model", \(model));
            const resp = await fetch("\(url)", {
              method: "POST",
              headers: { Authorization: "Bearer \(key)" },
              body: form,
            });
            console.log(await resp.json());
            """
        case .speech:
            return """
            import { writeFile } from "node:fs/promises";

            const resp = await fetch("\(url)", {
              method: "POST",
              headers: {
                Authorization: "Bearer \(key)",
                "Content-Type": "application/json",
              },
              body: JSON.stringify({ model: \(model), input: "Hello", voice: "Tingting" }),
            });
            await writeFile("speech.wav", Buffer.from(await resp.arrayBuffer()));
            """
        case .threed:
            return """
            const headers = {
              Authorization: "Bearer \(key)",
              "Content-Type": "application/json",
            };
            const created = await fetch("\(url)", {
              method: "POST",
              headers,
              body: JSON.stringify({ model: \(model), image: "/path/to/photo.png" }),
            });
            const job = await created.json();
            console.log(job);
            const status = await fetch(`\(url)/${job.id}`, { headers });
            console.log(await status.json());
            """
        default:
            return """
            const resp = await fetch("\(url)", {
              method: "POST",
              headers: {
                Authorization: "Bearer \(key)",
                "Content-Type": "application/json",
              },
              body: JSON.stringify(\(body(kind: kind, model: model))),
            });
            console.log(await resp.json());
            """
        }
    }
}

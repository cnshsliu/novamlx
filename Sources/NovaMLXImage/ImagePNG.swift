import CoreGraphics
import Foundation
import ImageIO
import NovaMLXCore
import UniformTypeIdentifiers

enum ImagePNG {
    static func cgImage(fromPNG data: Data) throws -> CGImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw NovaMLXError.inferenceFailed("Failed to decode PNG image data")
        }
        return image
    }
}

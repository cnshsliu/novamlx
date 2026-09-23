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

    static func pngData(from image: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data as CFMutableData, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw NovaMLXError.inferenceFailed("Failed to create PNG destination")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw NovaMLXError.inferenceFailed("Failed to encode PNG")
        }
        return data as Data
    }
}

import Testing
import Foundation
import NovaMLXCore

@Suite("ServerConfig allowUnlistedDownloads")
struct ServerConfigAllowUnlistedTests {
    @Test("Defaults to false")
    func defaultOff() {
        #expect(ServerConfig().allowUnlistedDownloads == false)
    }

    @Test("Legacy JSON without the key decodes as false")
    func legacyJSON() throws {
        let json = Data(#"{ "host": "127.0.0.1", "port": 6590, "adminPort": 6591 }"#.utf8)
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: json)
        #expect(cfg.allowUnlistedDownloads == false)
    }

    @Test("Decodes true")
    func decodesTrue() throws {
        let json = Data(#"{ "host": "127.0.0.1", "port": 6590, "adminPort": 6591, "allowUnlistedDownloads": true }"#.utf8)
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: json)
        #expect(cfg.allowUnlistedDownloads == true)
    }
}

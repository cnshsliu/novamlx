import XCTest
@testable import NovaMLXAPI
import NovaMLXCore

/// Mapper, request-parser, and route-catalog tests. No model and no GPU:
/// the HTTP handlers call `UnlimitedOCRMarkup` and `VendorOCRResponses`
/// with the engine string, and these tests feed that string in directly.
final class VendorOCRTests: XCTestCase {

    private let tinyPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

    private var tinyPNG: Data {
        Data(base64Encoded: tinyPNGBase64)!
    }

    // MARK: - Route registration

    func testRouteRegistrationSmoke() {
        let paths = VendorOCRCatalog.routes.map(\.path)
        XCTAssertEqual(paths, [
            "/rest/2.0/ocr/v1/general_basic",
            "/rest/2.0/ocr/v1/accurate_basic",
            "/rest/2.0/ocr/v1/general",
            "/rest/2.0/ocr/v1/accurate",
            "/ocr/RecognizeGeneral",
            "/ocr/GeneralBasicOCR",
            "/ocr/GeneralAccurateOCR",
        ])
        XCTAssertEqual(
            Set(VendorOCRCatalog.routes.map(\.vendor)),
            Set([VendorOCRVendor.baidu, .aliyun, .tencent])
        )

        for route in VendorOCRCatalog.routes {
            XCTAssertTrue(VendorOCRCatalog.isVendorOCRFamily(route.path))
            XCTAssertTrue(VendorOCRCatalog.matchesInferencePath(route.path))
            XCTAssertEqual(VendorOCRCatalog.route(path: route.path, action: nil), route)
            XCTAssertEqual(VendorOCRCatalog.route(path: route.path + "/", action: nil), route)
        }

        XCTAssertEqual(
            VendorOCRCatalog.route(path: "/ocr", action: "RecognizeAdvanced")?.path,
            VendorOCRCatalog.recognizeGeneral.path
        )
        XCTAssertEqual(
            VendorOCRCatalog.route(path: "/ocr/", action: "generalaccurateocr")?.product,
            .generalAccurateOCR
        )
        XCTAssertEqual(
            VendorOCRCatalog.route(path: "/ocr", action: "GeneralBasicOCR")?.vendor,
            .tencent
        )

        XCTAssertNil(VendorOCRCatalog.route(path: "/rest/2.0/ocr/v1/idcard", action: nil))
        XCTAssertNil(VendorOCRCatalog.route(path: "/ocr/IDCardOCR", action: nil))
        XCTAssertNil(VendorOCRCatalog.route(path: "/ocr/RecognizeIdcard", action: nil))
        XCTAssertTrue(VendorOCRCatalog.isVendorOCRFamily("/rest/2.0/ocr/v1/idcard"))
        XCTAssertEqual(
            VendorOCRCatalog.vendorForUnknown(path: "/rest/2.0/ocr/v1/idcard", action: nil),
            .baidu
        )
        XCTAssertEqual(
            VendorOCRCatalog.vendorForUnknown(path: "/ocr/RecognizeIdcard", action: nil),
            .aliyun
        )
        XCTAssertEqual(
            VendorOCRCatalog.vendorForUnknown(path: "/ocr/BankCardOCR", action: nil),
            .tencent
        )
        XCTAssertFalse(VendorOCRCatalog.matchesInferencePath("/v1/chat/completions"))
    }

    // MARK: - Engine prompt

    func testEnginePromptMatchesDocumentParsing() {
        XCTAssertEqual(VendorOCREngine.prompt, "document parsing.")
        XCTAssertEqual(VendorOCREngine.defaultModelID, "baidu/Unlimited-OCR-2")
        let message = VendorOCREngine.userMessage(image: "data:image/png;base64,abc")
        XCTAssertEqual(message.role, .user)
        XCTAssertEqual(message.content, "document parsing.")
        XCTAssertEqual(message.images, ["data:image/png;base64,abc"])
        let sampling = VendorOCREngine.sampling(modelID: "baidu/Unlimited-OCR-2")
        XCTAssertEqual(sampling.temperature, 0)
        XCTAssertEqual(sampling.maxTokens, 8192)
        XCTAssertNil(sampling.repetitionPenalty)
    }

    // MARK: - Markup

    func testParsesUnlimitedOCRMarkupIntoLinesAndBoxes() {
        let lines = UnlimitedOCRMarkup.parse(Self.invoiceMarkup)
        XCTAssertEqual(lines.map(\.text), [
            "INVOICE",
            "Date: 2024-01-01",
            "Widget | 2",
            "Cable | 1",
            "Second page",
        ])
        XCTAssertFalse(lines.contains { $0.text.contains("<") })
        XCTAssertEqual(lines[0].left, 132)
        XCTAssertEqual(lines[0].top, 45)
        XCTAssertEqual(lines[0].width, 728)
        XCTAssertEqual(lines[0].height, 53)
        XCTAssertEqual(lines[0].polygon[0], UnlimitedOCRPoint(x: 132, y: 45))
        XCTAssertEqual(lines[0].polygon[2], UnlimitedOCRPoint(x: 860, y: 98))
        XCTAssertEqual(lines[1].left, 50)
        XCTAssertEqual(lines[1].top, 120)
        XCTAssertEqual(lines[1].width, 350)
        XCTAssertEqual(lines[1].height, 40)
        XCTAssertEqual(lines[2].left, 40)
        XCTAssertEqual(lines[2].width, 860)
        XCTAssertEqual(lines[2].height, 200)
        XCTAssertEqual(lines[3].left, lines[2].left)
        XCTAssertEqual(lines[3].polygon, lines[2].polygon)
        XCTAssertEqual(lines[4].left, 10)
        XCTAssertEqual(lines[4].top, 10)
        XCTAssertEqual(lines[4].width, 10)
        XCTAssertEqual(lines[4].height, 20)
    }

    func testParsesPlainLinesWithoutBoxes() {
        let lines = UnlimitedOCRMarkup.parse("Alpha\n\nBeta\n")
        XCTAssertEqual(lines.map(\.text), ["Alpha", "Beta"])
        XCTAssertFalse(lines[0].hasBox)
        XCTAssertTrue(lines[0].polygon.isEmpty)
    }

    func testParsesSameLineDetAndQuad() {
        let same = UnlimitedOCRMarkup.parse("<|det|>[[1, 2, 30, 40]]<|/det|>Hello")
        XCTAssertEqual(same.count, 1)
        XCTAssertEqual(same[0].text, "Hello")
        XCTAssertEqual(same[0].left, 1)
        XCTAssertEqual(same[0].top, 2)
        XCTAssertEqual(same[0].width, 29)
        XCTAssertEqual(same[0].height, 38)

        let quad = UnlimitedOCRMarkup.parse("<|det|>[[0, 10, 8, 10, 8, 20, 0, 20]]<|/det|>Quad")
        XCTAssertEqual(quad[0].text, "Quad")
        XCTAssertEqual(quad[0].polygon.count, 4)
        XCTAssertEqual(quad[0].left, 0)
        XCTAssertEqual(quad[0].top, 10)
        XCTAssertEqual(quad[0].width, 8)
        XCTAssertEqual(quad[0].height, 10)
    }

    func testScalesNormalizedBoxesAndLeavesPixelBoxes() {
        let scaled = UnlimitedOCRMarkup.parse(
            "<|det|>[[0, 0, 999, 999]]<|/det|>Full",
            imagePixelSize: OCRPixelSize(width: 200, height: 100)
        )
        XCTAssertEqual(scaled[0].left, 0)
        XCTAssertEqual(scaled[0].top, 0)
        XCTAssertEqual(scaled[0].width, 200)
        XCTAssertEqual(scaled[0].height, 100)

        let pixels = UnlimitedOCRMarkup.parse(
            "<|det|>[[0, 0, 1500, 40]]<|/det|>Wide",
            imagePixelSize: OCRPixelSize(width: 2000, height: 1000)
        )
        XCTAssertEqual(pixels[0].width, 1500)
        XCTAssertEqual(pixels[0].height, 40)
    }

    // MARK: - Vendor JSON

    func testBaiduPayloadShape() {
        let lines = UnlimitedOCRMarkup.parse(Self.invoiceMarkup)
        let payload = VendorOCRResponses.success(
            vendor: .baidu, lines: lines, logID: 42, requestID: "req", imageSize: nil
        )
        XCTAssertEqual(payload.status, 200)
        let json = try? object(payload)
        XCTAssertEqual(json?["log_id"] as? Int, 42)
        XCTAssertEqual(json?["words_result_num"] as? Int, 5)
        XCTAssertNil(json?["error_code"])
        let words = json?["words_result"] as? [[String: Any]]
        XCTAssertEqual(words?.count, 5)
        XCTAssertEqual(words?[0]["words"] as? String, "INVOICE")
        let location = words?[0]["location"] as? [String: Any]
        XCTAssertEqual(location?["left"] as? Int, 132)
        XCTAssertEqual(location?["top"] as? Int, 45)
        XCTAssertEqual(location?["width"] as? Int, 728)
        XCTAssertEqual(location?["height"] as? Int, 53)

        let plain = UnlimitedOCRMarkup.parse("Only text")
        let plainPayload = VendorOCRResponses.success(
            vendor: .baidu, lines: plain, logID: 7, requestID: "req", imageSize: nil
        )
        let plainJSON = try? object(plainPayload)
        let plainWords = plainJSON?["words_result"] as? [[String: Any]]
        XCTAssertEqual(plainWords?[0]["words"] as? String, "Only text")
        XCTAssertNil(plainWords?[0]["location"])
    }

    func testAliyunPayloadShape() {
        let lines = UnlimitedOCRMarkup.parse("Hello")
        let payload = VendorOCRResponses.success(
            vendor: .aliyun,
            lines: lines,
            logID: 1,
            requestID: "req-ali",
            imageSize: OCRPixelSize(width: 640, height: 480)
        )
        XCTAssertEqual(payload.status, 200)
        let json = try? object(payload)
        XCTAssertEqual(json?["RequestId"] as? String, "req-ali")
        XCTAssertNil(json?["Code"])
        let dataField = json?["Data"] as? String
        let inner = dataField.flatMap { $0.data(using: .utf8) }.flatMap { data in
            try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        }
        XCTAssertEqual(inner?["content"] as? String, "Hello")
        XCTAssertEqual(inner?["prism_wnum"] as? Int, 1)
        XCTAssertEqual(inner?["width"] as? Int, 640)
        XCTAssertEqual(inner?["orgHeight"] as? Int, 480)
        let words = inner?["prism_wordsInfo"] as? [[String: Any]]
        XCTAssertEqual(words?[0]["word"] as? String, "Hello")
        XCTAssertEqual(words?[0]["prob"] as? Int, 0)
        let pos = words?[0]["pos"] as? [[String: Any]]
        XCTAssertEqual(pos?.count, 4)
        XCTAssertNotNil(pos?[0]["x"])
        XCTAssertNotNil(pos?[0]["y"])
    }

    func testAliyunUsesParsedBoxes() {
        let lines = UnlimitedOCRMarkup.parse("<|det|>[[10, 20, 30, 50]]<|/det|>Cell")
        let payload = VendorOCRResponses.success(
            vendor: .aliyun, lines: lines, logID: 1, requestID: "r", imageSize: nil
        )
        let json = try? object(payload)
        let dataField = json?["Data"] as? String
        let inner = dataField.flatMap { $0.data(using: .utf8) }.flatMap { data in
            try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        }
        let word = (inner?["prism_wordsInfo"] as? [[String: Any]])?.first
        XCTAssertEqual(word?["x"] as? Int, 10)
        XCTAssertEqual(word?["y"] as? Int, 20)
        XCTAssertEqual(word?["width"] as? Int, 20)
        XCTAssertEqual(word?["height"] as? Int, 30)
        let pos = word?["pos"] as? [[String: Any]]
        XCTAssertEqual(pos?[0]["x"] as? Int, 10)
        XCTAssertEqual(pos?[0]["y"] as? Int, 20)
        XCTAssertEqual(pos?[2]["x"] as? Int, 30)
        XCTAssertEqual(pos?[2]["y"] as? Int, 50)
    }

    func testTencentPayloadShape() {
        let lines = UnlimitedOCRMarkup.parse("<|det|>[[4, 6, 14, 16]]<|/det|>Detected")
        let payload = VendorOCRResponses.success(
            vendor: .tencent, lines: lines, logID: 1, requestID: "req-tx", imageSize: nil
        )
        XCTAssertEqual(payload.status, 200)
        let json = try? object(payload)
        let response = json?["Response"] as? [String: Any]
        XCTAssertNil(response?["Error"])
        XCTAssertEqual(response?["RequestId"] as? String, "req-tx")
        XCTAssertEqual(response?["Angel"] as? Int, 0)
        XCTAssertEqual(response?["Language"] as? String, "")
        let detections = response?["TextDetections"] as? [[String: Any]]
        XCTAssertEqual(detections?[0]["DetectedText"] as? String, "Detected")
        XCTAssertEqual(detections?[0]["Confidence"] as? Int, 0)
        XCTAssertEqual(detections?[0]["AdvancedInfo"] as? String, "{}")
        let polygon = detections?[0]["Polygon"] as? [[String: Any]]
        XCTAssertEqual(polygon?.count, 4)
        XCTAssertEqual(polygon?[0]["X"] as? Int, 4)
        XCTAssertEqual(polygon?[0]["Y"] as? Int, 6)
        XCTAssertEqual(polygon?[1]["X"] as? Int, 14)
        let item = detections?[0]["ItemPolygon"] as? [String: Any]
        XCTAssertEqual(item?["X"] as? Int, 4)
        XCTAssertEqual(item?["Y"] as? Int, 6)
        XCTAssertEqual(item?["Width"] as? Int, 10)
        XCTAssertEqual(item?["Height"] as? Int, 10)
    }

    func testEmptyMarkupIsAnEmptySuccess() {
        let lines = UnlimitedOCRMarkup.parse("   \n<PAGE>\n")
        XCTAssertTrue(lines.isEmpty)
        for vendor in [VendorOCRVendor.baidu, .aliyun, .tencent] {
            let payload = VendorOCRResponses.success(
                vendor: vendor, lines: lines, logID: 1, requestID: "r", imageSize: nil
            )
            XCTAssertEqual(payload.status, 200)
        }
        let baidu = try? object(VendorOCRResponses.success(
            vendor: .baidu, lines: [], logID: 9, requestID: "r", imageSize: nil
        ))
        XCTAssertEqual(baidu?["words_result_num"] as? Int, 0)
        XCTAssertEqual((baidu?["words_result"] as? [Any])?.count, 0)
    }

    func testVendorErrorShapes() {
        let missing = VendorOCRFailure.missingImage
        let baidu = VendorOCRResponses.failure(missing, vendor: .baidu, logID: 11, requestID: "r")
        XCTAssertEqual(baidu.status, 200)
        let baiduJSON = try? object(baidu)
        XCTAssertEqual(baiduJSON?["error_code"] as? Int, BaiduOCRErrorCode.missingParameter)
        XCTAssertEqual(baiduJSON?["error_msg"] as? String, "image or url is required")
        XCTAssertNil(baiduJSON?["words_result"])

        let format = VendorOCRResponses.failure(
            .imageFormat("image bytes are not a png, jpeg, gif, webp, or pdf"),
            vendor: .baidu, logID: 1, requestID: "r"
        )
        XCTAssertEqual((try? object(format))?["error_code"] as? Int, BaiduOCRErrorCode.imageFormat)

        let aliyun = VendorOCRResponses.failure(missing, vendor: .aliyun, logID: 1, requestID: "ali-1")
        XCTAssertEqual(aliyun.status, 400)
        let aliyunJSON = try? object(aliyun)
        XCTAssertEqual(aliyunJSON?["Code"] as? String, "MissingImage")
        XCTAssertEqual(aliyunJSON?["RequestId"] as? String, "ali-1")
        XCTAssertNotNil(aliyunJSON?["Message"])

        let down = VendorOCRResponses.failure(
            .unavailable("Model 'baidu/Unlimited-OCR-2' is not loaded."),
            vendor: .aliyun, logID: 1, requestID: "r"
        )
        XCTAssertEqual(down.status, 503)
        XCTAssertEqual((try? object(down))?["Code"] as? String, "ServiceUnavailable")

        let crashed = VendorOCRResponses.failure(.failed("inference failed"), vendor: .aliyun, logID: 1, requestID: "r")
        XCTAssertEqual(crashed.status, 500)

        let tencent = VendorOCRResponses.failure(
            .unsupported(VendorOCRCatalog.unsupportedMessage),
            vendor: .tencent, logID: 1, requestID: "tx-1"
        )
        XCTAssertEqual(tencent.status, 200)
        let response = (try? object(tencent))?["Response"] as? [String: Any]
        let error = response?["Error"] as? [String: Any]
        XCTAssertEqual(error?["Code"] as? String, "InvalidAction")
        XCTAssertTrue((error?["Message"] as? String)?.contains("out of scope") == true)
        XCTAssertEqual(response?["RequestId"] as? String, "tx-1")
        XCTAssertNil(response?["TextDetections"])
    }

    func testNovaErrorsMapOntoVendorFailures() {
        let missing = VendorOCRFailure.from(NovaMLXError.modelNotLoaded("baidu/Unlimited-OCR-2"))
        guard case .unavailable(let message) = missing else {
            return XCTFail("expected unavailable, got \(missing)")
        }
        XCTAssertTrue(message.contains("not loaded"))
        let payload = VendorOCRResponses.failure(missing, vendor: .baidu, logID: 3, requestID: "r")
        XCTAssertEqual((try? object(payload))?["error_code"] as? Int, BaiduOCRErrorCode.internalError)

        let bad = VendorOCRFailure.from(VendorOCRParseError.missingImage)
        XCTAssertEqual(bad, .missingImage)
        let route = VendorOCRFailure.from(VendorOCRRouteError.unsupported("moved"))
        XCTAssertEqual(route, .unsupported("moved"))
        let moved = VendorOCRFailure.from(VendorOCRRouteError.unavailable("gone"))
        XCTAssertEqual(moved, .unavailable("gone"))
    }

    // MARK: - Requests

    func testFormBodyDecodesBase64AndIgnoresAccessToken() throws {
        let encoded = tinyPNG.base64EncodedString()
            .addingPercentEncoding(withAllowedCharacters: .alphanumerics)!
        let body = Data("image=\(encoded)&access_token=baidu-token".utf8)
        let parsed = try VendorOCRRequestParser.parse(
            body: body,
            contentType: "application/x-www-form-urlencoded"
        )
        guard case .dataURL(let url, let size) = parsed.image else {
            return XCTFail("expected a data URL, got \(parsed.image)")
        }
        XCTAssertTrue(url.hasPrefix("data:image/png;base64,"))
        let payload = url.split(separator: ",", maxSplits: 1).last.map(String.init) ?? ""
        XCTAssertEqual(Data(base64Encoded: payload), tinyPNG)
        XCTAssertEqual(size, OCRPixelSize(width: 1, height: 1))
        XCTAssertEqual(parsed.modelID, "baidu/Unlimited-OCR-2")
        XCTAssertFalse(url.contains("baidu-token"))
    }

    func testJSONAcceptsVendorURLAndImageFields() throws {
        let aliyun = try VendorOCRRequestParser.parse(
            body: Data(#"{"Url":"https://example.com/a.png","SecretId":"not-a-key"}"#.utf8),
            contentType: "application/json"
        )
        guard case .remoteURL(let url) = aliyun.image else {
            return XCTFail("expected a remote URL")
        }
        XCTAssertEqual(url, "https://example.com/a.png")
        XCTAssertEqual(aliyun.modelID, VendorOCREngine.defaultModelID)

        let tencent = try VendorOCRRequestParser.parse(
            body: Data(#"{"ImageUrl":"https://example.com/b.png","model":"mlx-community/Unlimited-OCR-8bit"}"#.utf8),
            contentType: "application/json"
        )
        guard case .remoteURL(let imageURL) = tencent.image else {
            return XCTFail("expected a remote URL")
        }
        XCTAssertEqual(imageURL, "https://example.com/b.png")
        XCTAssertEqual(tencent.modelID, "mlx-community/Unlimited-OCR-8bit")

        let both = try VendorOCRRequestParser.parse(
            body: Data("""
            {"image":"\(tinyPNGBase64)","url":"https://example.com/ignored.png"}
            """.utf8),
            contentType: "application/json"
        )
        guard case .dataURL = both.image else {
            return XCTFail("image bytes win over url")
        }
    }

    func testHeaderModelOverridesBody() throws {
        let parsed = try VendorOCRRequestParser.parse(
            body: Data(#"{"url":"https://example.com/a.png","model":"other"}"#.utf8),
            contentType: "application/json",
            query: ["model": "from-query"],
            headerModel: "baidu/Unlimited-OCR-2"
        )
        XCTAssertEqual(parsed.modelID, "baidu/Unlimited-OCR-2")
    }

    func testQueryURLFillsAnEmptyBody() throws {
        let parsed = try VendorOCRRequestParser.parse(
            body: Data("{}".utf8),
            contentType: "application/json",
            query: ["url": "https://example.com/q.png", "access_token": "ignored"]
        )
        guard case .remoteURL(let url) = parsed.image else {
            return XCTFail("expected the query url")
        }
        XCTAssertEqual(url, "https://example.com/q.png")
    }

    func testMissingAndInvalidImages() {
        XCTAssertThrowsError(try VendorOCRRequestParser.parse(
            body: Data(#"{"language_type":"CHN_ENG"}"#.utf8),
            contentType: "application/json"
        )) { error in
            XCTAssertEqual(error as? VendorOCRParseError, .missingImage)
        }
        XCTAssertThrowsError(try VendorOCRRequestParser.parse(
            body: Data(#"{"image":"this is not base64!!!"}"#.utf8),
            contentType: "application/json"
        )) { error in
            guard case .imageFormat = error as? VendorOCRParseError else {
                return XCTFail("expected imageFormat, got \(error)")
            }
        }
    }

    func testPDFBase64AndRawPNG() throws {
        let pdf = Data("%PDF-1.4\n".utf8)
        let body = Data("""
        {"pdf_file":"\(pdf.base64EncodedString())"}
        """.utf8)
        let parsed = try VendorOCRRequestParser.parse(body: body, contentType: "application/json")
        guard case .pdf(let data) = parsed.image else {
            return XCTFail("expected pdf bytes")
        }
        XCTAssertTrue(data.starts(with: Data("%PDF".utf8)))

        let raw = try VendorOCRRequestParser.parse(body: tinyPNG, contentType: "image/png")
        guard case .dataURL(_, let size) = raw.image else {
            return XCTFail("expected a data URL")
        }
        XCTAssertEqual(size, OCRPixelSize(width: 1, height: 1))
    }

    func testMultipartFileUpload() throws {
        let boundary = "NovaOCRBoundary"
        var body = Data()
        func add(_ string: String) { body.append(Data(string.utf8)) }
        add("--\(boundary)\r\n")
        add("Content-Disposition: form-data; name=\"image\"; filename=\"page.png\"\r\n")
        add("Content-Type: image/png\r\n\r\n")
        body.append(tinyPNG)
        add("\r\n--\(boundary)\r\n")
        add("Content-Disposition: form-data; name=\"model\"\r\n\r\n")
        add("mlx-community/Unlimited-OCR-8bit")
        add("\r\n--\(boundary)--\r\n")

        let parsed = try VendorOCRRequestParser.parse(
            body: body,
            contentType: "multipart/form-data; boundary=\(boundary)"
        )
        guard case .dataURL(_, let size) = parsed.image else {
            return XCTFail("expected the uploaded png")
        }
        XCTAssertEqual(size, OCRPixelSize(width: 1, height: 1))
        XCTAssertEqual(parsed.modelID, "mlx-community/Unlimited-OCR-8bit")
    }

    // MARK: - Helpers

    private func object(_ payload: VendorOCRPayload) throws -> [String: Any] {
        let data = VendorOCRResponses.encode(payload)
        let json = try JSONSerialization.jsonObject(with: data)
        return try XCTUnwrap(json as? [String: Any])
    }

    private static let invoiceMarkup = """
    <PAGE>
    <|ref|>title<|/ref|><|det|>[[132, 45, 860, 98]]<|/det|>
    # INVOICE
    <|ref|>text<|/ref|><|det|>[[50, 120, 400, 160]]<|/det|>
    Date: 2024-01-01
    <|ref|>table<|/ref|><|det|>[[40, 200, 900, 400]]<|/det|>
    <table><tr><td>Widget</td><td>2</td></tr><tr><td>Cable</td><td>1</td></tr></table>
    <PAGE>
    <|det|>[[10, 10, 20, 30]]<|/det|>Second page
    """
}

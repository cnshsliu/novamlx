// Copyright © 2024 Apple Inc.
// Adapted for NovaMLX: removed Hub dependency, ModelContainer removed.

import Foundation
import MLX
import MLXNN

/// Iterator that produces latent images.
public struct DenoiseIterator: Sequence, IteratorProtocol {

    let sd: StableDiffusion

    var xt: MLXArray

    let conditioning: MLXArray
    let cfgWeight: Float
    let textTime: (MLXArray, MLXArray)?

    var i: Int
    let steps: [(MLXArray, MLXArray)]

    init(
        sd: StableDiffusion, xt: MLXArray, t: Int, conditioning: MLXArray, steps: Int,
        cfgWeight: Float, textTime: (MLXArray, MLXArray)? = nil
    ) {
        self.sd = sd
        self.steps = sd.sampler.timeSteps(steps: steps, start: t, dType: sd.dType)
        self.i = 0
        self.xt = xt
        self.conditioning = conditioning
        self.cfgWeight = cfgWeight
        self.textTime = textTime
    }

    public var underestimatedCount: Int {
        steps.count
    }

    mutating public func next() -> MLXArray? {
        guard i < steps.count else {
            return nil
        }

        let (t, tPrev) = steps[i]
        i += 1

        xt = sd.step(
            xt: xt, t: t, tPrev: tPrev, conditioning: conditioning, cfgWeight: cfgWeight,
            textTime: textTime)
        return xt
    }
}

/// Type for the _decoder_ step.
public typealias ImageDecoder = (MLXArray) -> MLXArray

public protocol ImageGenerator {
    func ensureLoaded()
    func detachedDecoder() -> ImageDecoder
    func decode(xt: MLXArray) -> MLXArray
}

/// Public interface for transforming a text prompt into an image.
public protocol TextToImageGenerator: ImageGenerator {
    func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator
}

/// Public interface for image-to-image generation.
public protocol ImageToImageGenerator: ImageGenerator {
    func generateLatents(image: MLXArray, parameters: EvaluateParameters, strength: Float)
        -> DenoiseIterator
}

/// Base class for Stable Diffusion.
open class StableDiffusion {

    let dType: DType
    let diffusionConfiguration: DiffusionConfiguration
    let unet: UNetModel
    let textEncoder: CLIPTextModel
    let autoencoder: Autoencoder
    let sampler: SimpleEulerSampler
    let tokenizer: CLIPTokenizer

    init(
        directory: URL, configuration: StableDiffusionConfiguration, dType: DType,
        diffusionConfiguration: DiffusionConfiguration? = nil, unet: UNetModel? = nil,
        textEncoder: CLIPTextModel? = nil, autoencoder: Autoencoder? = nil,
        sampler: SimpleEulerSampler? = nil, tokenizer: CLIPTokenizer? = nil
    ) throws {
        self.dType = dType
        self.diffusionConfiguration =
            try diffusionConfiguration
            ?? loadDiffusionConfiguration(directory: directory, configuration: configuration)
        self.unet = try unet ?? loadUnet(
            directory: directory, configuration: configuration, dType: dType)
        self.textEncoder =
            try textEncoder ?? loadTextEncoder(
                directory: directory, configuration: configuration, dType: dType)
        self.autoencoder =
            try autoencoder
            ?? loadAutoEncoder(directory: directory, configuration: configuration, dType: .float32)

        if let sampler {
            self.sampler = sampler
        } else {
            self.sampler = SimpleEulerSampler(configuration: self.diffusionConfiguration)
        }
        self.tokenizer = try tokenizer ?? loadTokenizer(
            directory: directory, configuration: configuration)
    }

    open func ensureLoaded() {
        // Evaluate individual arrays within each leaf module rather than the
        // whole module at once.  `eval(leaf)` gathers *all* parameters of the
        // leaf into a single vector and submits them in one batch; for a large
        // Linear or Conv2d that can still be hundreds of MB and exceed the
        // macOS background-process command-buffer timeout.  Evaluating one
        // MLXArray at a time keeps each submission tiny.
        for (_, leaf) in unet.leafModules().flattened() {
            for (_, arr) in leaf.parameters().flattened() {
                eval(arr)
            }
        }
        for (_, leaf) in textEncoder.leafModules().flattened() {
            for (_, arr) in leaf.parameters().flattened() {
                eval(arr)
            }
        }
        for (_, leaf) in autoencoder.leafModules().flattened() {
            for (_, arr) in leaf.parameters().flattened() {
                eval(arr)
            }
        }
    }

    func tokenize(tokenizer: CLIPTokenizer, text: String, negativeText: String?) -> MLXArray {
        var tokens = [tokenizer.tokenize(text: text)]
        if let negativeText {
            tokens.append(tokenizer.tokenize(text: negativeText))
        }

        let c = tokens.count
        let max = tokens.map { $0.count }.max() ?? 0
        let mlxTokens = MLXArray(
            tokens
                .map {
                    ($0 + Array(repeating: 0, count: max - $0.count))
                }
                .flatMap { $0 }
        )
        .reshaped(c, max)

        return mlxTokens
    }

    open func step(
        xt: MLXArray, t: MLXArray, tPrev: MLXArray, conditioning: MLXArray, cfgWeight: Float,
        textTime: (MLXArray, MLXArray)?
    ) -> MLXArray {
        let xtUnet = cfgWeight > 1 ? concatenated([xt, xt], axis: 0) : xt
        let tUnet = broadcast(t, to: [xtUnet.count])

        var epsPred = unet(xtUnet, timestep: tUnet, encoderX: conditioning, textTime: textTime)
        // Eval UNet output immediately to keep GPU command buffers small.
        // Without this the UNet + CFG + sampler all land in one command buffer
        // that can exceed macOS's ~5s GPU timeout.
        eval(epsPred)

        if cfgWeight > 1 {
            let (epsText, epsNeg) = epsPred.split()
            epsPred = epsNeg + cfgWeight * (epsText - epsNeg)
        }

        return sampler.step(epsPred: epsPred, xt: xt, t: t, tPrev: tPrev)
    }

    public func detachedDecoder() -> ImageDecoder {
        let autoencoder = self.autoencoder
        func decode(xt: MLXArray) -> MLXArray {
            var x = autoencoder.decode(xt)
            x = clip(x / 2 + 0.5, min: 0, max: 1)
            return x
        }
        return decode(xt:)
    }

    public func decode(xt: MLXArray) -> MLXArray {
        detachedDecoder()(xt)
    }
}

/// Implementation for stable-diffusion-2-1-base.
open class StableDiffusionBase: StableDiffusion, TextToImageGenerator {

    public init(directory: URL, configuration: StableDiffusionConfiguration, dType: DType) throws {
        try super.init(directory: directory, configuration: configuration, dType: dType)
    }

    func conditionText(text: String, imageCount: Int, cfgWeight: Float, negativeText: String?)
        -> MLXArray
    {
        let tokens = tokenize(
            tokenizer: tokenizer, text: text, negativeText: cfgWeight > 1 ? negativeText : nil)
        var conditioning = textEncoder(tokens).lastHiddenState
        if imageCount > 1 {
            conditioning = repeated(conditioning, count: imageCount, axis: 0)
        }
        return conditioning
    }

    public func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator {
        MLXRandom.seed(parameters.seed)

        let conditioning = conditionText(
            text: parameters.prompt, imageCount: parameters.imageCount,
            cfgWeight: parameters.cfgWeight, negativeText: parameters.negativePrompt)

        eval(conditioning)

        let xt = sampler.samplePrior(
            shape: [parameters.imageCount] + parameters.latentSize + [autoencoder.latentChannels],
            dType: dType)
        eval(xt)

        return DenoiseIterator(
            sd: self, xt: xt, t: sampler.maxTime, conditioning: conditioning,
            steps: parameters.steps, cfgWeight: parameters.cfgWeight)
    }
}

/// Implementation for sdxl-turbo.
open class StableDiffusionXL: StableDiffusion, TextToImageGenerator, ImageToImageGenerator {

    let textEncoder2: CLIPTextModel
    let tokenizer2: CLIPTokenizer

    public init(directory: URL, configuration: StableDiffusionConfiguration, dType: DType) throws {
        let diffusionConfiguration = try loadConfiguration(
            directory: directory, configuration: configuration, key: .diffusionConfig,
            type: DiffusionConfiguration.self)
        let sampler = SimpleEulerAncestralSampler(configuration: diffusionConfiguration)

        self.textEncoder2 = try loadTextEncoder(
            directory: directory, configuration: configuration, configKey: .textEncoderConfig2,
            weightsKey: .textEncoderWeights2, dType: dType)

        self.tokenizer2 = try loadTokenizer(
            directory: directory, configuration: configuration, vocabulary: .tokenizerVocabulary2,
            merges: .tokenizerMerges2)

        try super.init(
            directory: directory, configuration: configuration, dType: dType,
            diffusionConfiguration: diffusionConfiguration, sampler: sampler)
    }

    open override func ensureLoaded() {
        super.ensureLoaded()
        for (_, leaf) in textEncoder2.leafModules().flattened() {
            for (_, arr) in leaf.parameters().flattened() {
                eval(arr)
            }
        }
    }

    func conditionText(text: String, imageCount: Int, cfgWeight: Float, negativeText: String?) -> (
        MLXArray, MLXArray
    ) {
        let tokens1 = tokenize(
            tokenizer: tokenizer, text: text, negativeText: cfgWeight > 1 ? negativeText : nil)
        let tokens2 = tokenize(
            tokenizer: tokenizer2, text: text, negativeText: cfgWeight > 1 ? negativeText : nil)

        let conditioning1 = textEncoder(tokens1)
        let conditioning2 = textEncoder2(tokens2)
        var conditioning = concatenated(
            [
                conditioning1.hiddenStates.dropLast().last!,
                conditioning2.hiddenStates.dropLast().last!,
            ],
            axis: -1)
        var pooledConditionng = conditioning2.pooledOutput

        if imageCount > 1 {
            conditioning = repeated(conditioning, count: imageCount, axis: 0)
            pooledConditionng = repeated(pooledConditionng, count: imageCount, axis: 0)
        }

        return (conditioning, pooledConditionng)
    }

    public func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator {
        MLXRandom.seed(parameters.seed)

        let (conditioning, pooledConditioning) = conditionText(
            text: parameters.prompt, imageCount: parameters.imageCount,
            cfgWeight: parameters.cfgWeight, negativeText: parameters.negativePrompt)

        // Eagerly eval text conditioning to keep GPU command buffers small.
        // Without this, the entire text encoder + first UNet forward pass
        // gets materialized in a single command buffer that can exceed
        // macOS's ~5s GPU timeout for background processes.
        eval(conditioning, pooledConditioning)

        let textTime = (
            pooledConditioning,
            repeated(
                MLXArray(converting: [512.0, 512, 0, 0, 512, 512]).reshaped(1, -1),
                count: pooledConditioning.count, axis: 0)
        )

        let xt = sampler.samplePrior(
            shape: [parameters.imageCount] + parameters.latentSize + [autoencoder.latentChannels],
            dType: dType)
        eval(xt)

        return DenoiseIterator(
            sd: self, xt: xt, t: sampler.maxTime, conditioning: conditioning,
            steps: parameters.steps, cfgWeight: parameters.cfgWeight, textTime: textTime)
    }

    public func generateLatents(image: MLXArray, parameters: EvaluateParameters, strength: Float)
        -> DenoiseIterator
    {
        MLXRandom.seed(parameters.seed)

        let startStep = Float(sampler.maxTime) * strength
        let numSteps = Int(Float(parameters.steps) * strength)

        let (conditioning, pooledConditioning) = conditionText(
            text: parameters.prompt, imageCount: parameters.imageCount,
            cfgWeight: parameters.cfgWeight, negativeText: parameters.negativePrompt)

        eval(conditioning, pooledConditioning)

        let textTime = (
            pooledConditioning,
            repeated(
                MLXArray(converting: [512.0, 512, 0, 0, 512, 512]).reshaped(1, -1),
                count: pooledConditioning.count, axis: 0)
        )

        var (x0, _) = autoencoder.encode(image[.newAxis])
        eval(x0)
        x0 = broadcast(x0, to: [parameters.imageCount] + x0.shape.dropFirst())
        let xt = sampler.addNoise(x: x0, t: MLXArray(startStep))
        eval(xt)

        return DenoiseIterator(
            sd: self, xt: xt, t: sampler.maxTime, conditioning: conditioning, steps: numSteps,
            cfgWeight: parameters.cfgWeight, textTime: textTime)
    }
}

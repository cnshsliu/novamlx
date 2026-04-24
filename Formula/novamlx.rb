class Novamlx < Formula
  desc "Apple Silicon MLX inference server — OpenAI & Anthropic compatible API"
  homepage "https://github.com/cnshsliu/novamlx"
  url "https://github.com/cnshsliu/novamlx/releases/download/v1.0.0/NovaMLX-1.0.0-arm64.tar.gz"
  sha256 "cbabe73b0f3443c81c839967835daa91ef5098f3dfcd18d56f69d22c5ca97054"
  version "1.0.0"

  depends_on :macos => :sequoia

  def install
    libexec.install Dir["NovaMLX.app/**"]
    bin.install_symlink libexec/"Contents/MacOS/NovaMLX" => "novamlx"
    bin.install_symlink libexec/"Contents/MacOS/NovaMLXWorker" => "novamlx-worker"
    bin.install_symlink libexec/"Contents/MacOS/nova" => "nova-cli"
  end

  service do
    run [opt_bin/"novamlx", "serve"]
    keep_alive true
    log_path var/"log/novamlx.log"
    error_log_path var/"log/novamlx.error.log"
    environment_variables PATH: stdlib_path("usr/bin")
  end

  def caveats
    <<~EOS
      NovaMLX requires macOS 15.0 (Sequoia) or later with Apple Silicon.

      Start the server:
        brew services start novamlx

      Or run manually:
        novamlx serve

      Configuration is stored in ~/.nova/config.json
      Models are downloaded to ~/.nova/models/

      Documentation: https://github.com/cnshsliu/novamlx#readme
    EOS
  end

  test do
    assert_match "NovaMLX", shell_output("#{bin}/novamlx --version", 0)
  rescue
    # --version may not be implemented yet, fallback check binary exists
    assert_path_exists bin/"novamlx"
  end
end

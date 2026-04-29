import Foundation
import Hummingbird

enum WebUIBuilder {
    static func render() -> String {
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NovaMLX</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.1/marked.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
        <style>
        \(SharedHTML.css)
        </style>
        </head>
        <body>
        \(SharedHTML.navBar)
        \(StatusHTML.content)
        \(ModelsHTML.content)
        \(ChatPageHTML.fragment)
        \(AgentsHTML.content)
        \(SettingsHTML.content)
        <div id="apiKeyModal" class="modal-overlay" style="display:none">
            <div class="modal-card">
                <h3>API Key</h3>
                <input type="text" id="apiKeyInput" class="input-field" placeholder="Enter API key">
                <div class="modal-actions">
                    <button class="btn-secondary" onclick="hideApiKeyModal()">Cancel</button>
                    <button class="btn-primary" onclick="saveApiKey()">Save</button>
                </div>
            </div>
        </div>
        <div id="toastContainer"></div>
        <script>
        \(SharedHTML.js)
        \(StatusHTML.js)
        \(ModelsHTML.js)
        \(ChatPageHTML.js)
        \(AgentsHTML.js)
        \(SettingsHTML.js)
        window.StatusPage=StatusPage;
        window.ModelsPage=ModelsPage;
        window.ChatPage=ChatPage;
        window.AgentsPage=AgentsPage;
        window.SettingsPage=SettingsPage;
        window.addEventListener('load',()=>{const h=location.hash.slice(1)||'chat';navigateTo(h)});
        window.addEventListener('hashchange',()=>{navigateTo(location.hash.slice(1))});
        </script>
        </body>
        </html>
        """
    }
}

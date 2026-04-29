import Foundation

// Chat page fragment for the SPA.
// Phase 0: uses iframe pointing to the legacy /chat page.
// Phase 1: will be replaced with a proper inline fragment.
enum ChatPageHTML {
    static var fragment: String {
        """
        <div id="page-chat" class="page-container" style="display:none">
            <iframe src="/chat" style="width:100%;height:100%;border:none"></iframe>
        </div>
        """
    }

    // Placeholder JS — no-op until Phase 1 replaces the iframe
    static var js: String {
        """
        const ChatPage={
            activate(){},
            deactivate(){}
        };
        """
    }
}

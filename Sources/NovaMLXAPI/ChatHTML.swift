import Foundation
import Hummingbird

enum ChatHTML {
    static func render() -> String {
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NovaMLX Chat</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.1/marked.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        <style>
        *{margin:0;padding:0;box-sizing:border-box}
        :root{--bg:#0a0a0a;--bg2:#141414;--bg3:#1e1e1e;--bg4:#282828;--text:#e5e5e5;--text2:#a3a3a3;--text3:#737373;--border:#262626;--border2:#333;--accent:#8b5cf6;--accent2:#7c3aed;--green:#22c55e;--red:#ef4444;--yellow:#eab308;--blue:#3b82f6;--code-bg:#1a1a2e;--radius:8px}
        body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);height:100vh;overflow:hidden;display:flex}
        .sidebar{width:260px;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0}
        .sidebar-header{padding:16px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
        .sidebar-header h2{font-size:14px;font-weight:600;color:var(--text)}
        .sidebar-header a{color:var(--text3);text-decoration:none;font-size:12px}
        .sidebar-header a:hover{color:var(--text)}
        .new-chat-btn{margin:12px 16px;padding:8px 12px;background:var(--accent);color:#fff;border:none;border-radius:var(--radius);cursor:pointer;font-size:13px;font-weight:500}
        .new-chat-btn:hover{background:var(--accent2)}
        .search-box{margin:0 16px 8px;padding:6px 10px;background:var(--bg3);border:1px solid var(--border2);border-radius:6px;color:var(--text);font-size:12px;outline:none;width:calc(100% - 32px)}
        .search-box:focus{border-color:var(--accent)}
        .chat-list{flex:1;overflow-y:auto;padding:4px 8px}
        .chat-item{padding:10px 12px;border-radius:6px;cursor:pointer;font-size:13px;color:var(--text2);display:flex;align-items:center;gap:8px;margin-bottom:2px;position:relative}
        .chat-item:hover{background:var(--bg3);color:var(--text)}
        .chat-item.active{background:var(--bg3);color:var(--text)}
        .chat-item .delete-btn{position:absolute;right:8px;opacity:0;background:none;border:none;color:var(--red);cursor:pointer;font-size:12px;padding:2px 4px}
        .chat-item:hover .delete-btn{opacity:1}
        .chat-item .title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
        .sidebar-bottom{padding:12px 16px;border-top:1px solid var(--border);display:flex;flex-direction:column;gap:6px}
        .sidebar-bottom button{width:100%}
        .main{flex:1;display:flex;flex-direction:column;min-width:0}
        .topbar{padding:12px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;background:var(--bg2)}
        .topbar h1{font-size:16px;font-weight:600;flex:1}
        .topbar h1 span{color:var(--accent)}
        select{background:var(--bg3);color:var(--text);border:1px solid var(--border2);border-radius:6px;padding:6px 10px;font-size:13px;cursor:pointer;outline:none;max-width:250px}
        select:focus{border-color:var(--accent)}
        .topbar-btn{background:var(--bg3);border:1px solid var(--border2);color:var(--text2);padding:6px 10px;border-radius:6px;cursor:pointer;font-size:12px}
        .topbar-btn:hover{background:var(--bg4);color:var(--text)}
        .messages-wrapper{flex:1;position:relative;overflow:hidden}
        .messages{flex:1;overflow-y:auto;padding:20px;scroll-behavior:smooth;height:100%}
        .scroll-bottom-btn{position:absolute;bottom:20px;right:20px;width:36px;height:36px;border-radius:50%;background:var(--accent);color:#fff;border:none;cursor:pointer;font-size:16px;display:none;align-items:center;justify-content:center;z-index:10;box-shadow:0 2px 8px rgba(0,0,0,.4)}
        .scroll-bottom-btn.visible{display:flex}
        .msg{margin-bottom:16px;max-width:800px;margin-left:auto;margin-right:auto}
        .msg-header{font-size:11px;color:var(--text3);margin-bottom:4px;display:flex;align-items:center;gap:8px}
        .msg-header .role{font-weight:600;text-transform:uppercase}
        .msg-header .role.user{color:var(--blue)}
        .msg-header .role.assistant{color:var(--green)}
        .msg-header .role.system{color:var(--yellow)}
        .msg-header .ts{cursor:help}
        .msg-body{font-size:14px;line-height:1.7;color:var(--text);padding:12px 16px;border-radius:var(--radius);background:var(--bg2);border:1px solid var(--border);word-wrap:break-word;overflow-wrap:break-word}
        .msg-body.user-msg{background:var(--bg3)}
        .msg-body p{margin-bottom:8px}
        .msg-body p:last-child{margin-bottom:0}
        .msg-body code{background:var(--code-bg);padding:2px 6px;border-radius:4px;font-size:13px;font-family:"SF Mono",Monaco,"Cascadia Code",monospace}
        .msg-body pre{background:var(--code-bg);border-radius:6px;overflow:hidden;margin:8px 0;position:relative}
        .msg-body pre code{display:block;padding:12px 16px;font-size:13px;background:none;overflow-x:auto}
        .code-header{display:flex;justify-content:space-between;align-items:center;padding:4px 12px;background:var(--bg4);font-size:11px;color:var(--text3);font-family:monospace}
        .code-header .lang{font-weight:600;text-transform:uppercase}
        .code-header .copy-code-btn{background:none;border:none;color:var(--text3);cursor:pointer;font-size:11px;padding:2px 6px;border-radius:4px}
        .code-header .copy-code-btn:hover{color:var(--text);background:var(--bg3)}
        .msg-body ul,.msg-body ol{margin:8px 0;padding-left:24px}
        .msg-body li{margin-bottom:4px}
        .msg-body blockquote{border-left:3px solid var(--accent);padding-left:12px;color:var(--text2);margin:8px 0}
        .msg-body a{color:var(--accent)}
        .msg-body h1,.msg-body h2,.msg-body h3,.msg-body h4{margin:12px 0 8px;color:var(--text)}
        .msg-body h1{font-size:20px}.msg-body h2{font-size:17px}.msg-body h3{font-size:15px}.msg-body h4{font-size:14px}
        .msg-body table{border-collapse:collapse;width:100%;margin:8px 0}
        .msg-body th,.msg-body td{border:1px solid var(--border2);padding:6px 10px;text-align:left;font-size:13px}
        .msg-body th{background:var(--bg3)}
        .msg-body hr{border:none;border-top:1px solid var(--border2);margin:12px 0}
        .msg-body img{max-width:100%;border-radius:6px;cursor:pointer;margin:4px 0}
        .msg-body input[type="checkbox"]{margin-right:6px;accent-color:var(--accent)}
        .msg-actions{display:flex;gap:4px;margin-top:6px;opacity:0;transition:opacity .2s}
        .msg:hover .msg-actions{opacity:1}
        .msg-actions button{background:none;border:none;color:var(--text3);cursor:pointer;font-size:11px;padding:2px 6px;border-radius:4px}
        .msg-actions button:hover{background:var(--bg3);color:var(--text)}
        .thinking-bubble{margin-bottom:8px}
        .thinking-toggle{display:flex;align-items:center;gap:6px;cursor:pointer;padding:6px 10px;background:var(--bg3);border:1px solid var(--border);border-radius:6px;font-size:12px;color:var(--text3)}
        .thinking-toggle:hover{background:var(--bg4)}
        .thinking-content{padding:10px;margin-top:4px;background:var(--bg);border-radius:6px;font-size:13px;color:var(--text3);max-height:300px;overflow-y:auto;display:none}
        .thinking-content.open{display:block}
        .thinking-toggle .arrow{transition:transform .2s;font-size:10px}
        .thinking-toggle.open .arrow{transform:rotate(90deg)}
        .thought-summary{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;background:var(--bg3);border:1px solid var(--border);border-radius:14px;font-size:12px;color:var(--text3);cursor:pointer;margin-bottom:8px;transition:all .15s}
        .thought-summary:hover{background:var(--bg4);color:var(--text);border-color:var(--accent)}
        .thought-summary .dot{width:6px;height:6px;border-radius:50%;background:var(--accent);flex-shrink:0}
        .neural-pulse{margin-bottom:10px;cursor:pointer;border-radius:10px;overflow:hidden;border:1px solid var(--border);background:var(--bg2);transition:border-color .3s}
        .neural-pulse:hover{border-color:var(--accent)}
        .neural-pulse-bar{height:2px;background:linear-gradient(90deg,transparent,var(--accent),#c084fc,var(--accent),transparent);background-size:200% 100%;animation:np-slide 1.8s ease-in-out infinite}
        @keyframes np-slide{0%{background-position:-200% 0}100%{background-position:200% 0}}
        .neural-pulse-body{display:flex;align-items:center;gap:10px;padding:10px 14px}
        .neural-pulse-dot{width:8px;height:8px;border-radius:50%;background:var(--accent);animation:np-glow 1.4s ease-in-out infinite;flex-shrink:0;box-shadow:0 0 8px var(--accent)}
        @keyframes np-glow{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.45;transform:scale(.75)}}
        .neural-pulse-metrics{flex:1;display:flex;flex-direction:column;gap:2px;min-width:0}
        .neural-pulse-label{font-size:12px;font-weight:600;color:var(--text2)}
        .neural-pulse-stats{font-size:11px;color:var(--text3);font-variant-numeric:tabular-nums}
        .neural-pulse-ghost{font-size:11px;color:var(--text3);opacity:.35;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;max-width:180px;letter-spacing:.02em;font-style:italic;flex-shrink:0}
        .sync-indicator{margin-bottom:10px;border-radius:10px;overflow:hidden;border:1px solid var(--border);background:var(--bg2)}
        .sync-bar{height:2px;background:linear-gradient(90deg,transparent,var(--accent),#c084fc,var(--accent),transparent);background-size:200% 100%;animation:np-slide 1.8s ease-in-out infinite}
        .sync-body{display:flex;align-items:center;gap:10px;padding:10px 14px}
        .sync-dot{width:8px;height:8px;border-radius:50%;background:var(--accent);animation:np-glow 1.4s ease-in-out infinite;flex-shrink:0;box-shadow:0 0 8px var(--accent)}
        .sync-text{font-size:12px;color:var(--text2);font-weight:500}
        .sync-timer{font-size:11px;color:var(--text3);font-variant-numeric:tabular-nums}
        .typing-cursor{display:inline-block;width:2px;height:1em;background:var(--accent);margin-left:2px;vertical-align:text-bottom;animation:blink-cursor .8s step-end infinite}
        @keyframes blink-cursor{0%,100%{opacity:1}50%{opacity:0}}
        .thinking-panel{position:fixed;top:0;right:-420px;width:400px;height:100vh;background:var(--bg2);border-left:1px solid var(--border);z-index:500;transition:right .3s ease;display:flex;flex-direction:column}
        .thinking-panel.open{right:0}
        .thinking-panel-header{display:flex;justify-content:space-between;align-items:center;padding:16px 20px;border-bottom:1px solid var(--border)}
        .thinking-panel-header h3{font-size:14px;color:var(--text)}
        .thinking-panel-close{background:none;border:none;color:var(--text3);cursor:pointer;font-size:18px;padding:4px 8px;border-radius:4px}
        .thinking-panel-close:hover{background:var(--bg3);color:var(--text)}
        .thinking-panel-body{flex:1;overflow-y:auto;padding:16px 20px;font-size:13px;line-height:1.7;color:var(--text3)}
        .thinking-panel-body p{margin-bottom:8px}
        .thinking-panel-body code{background:var(--code-bg);padding:2px 6px;border-radius:4px;font-size:12px}
        .thinking-panel-body pre{background:var(--code-bg);border-radius:6px;padding:12px;overflow-x:auto;margin:8px 0}
        .thinking-overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.3);z-index:499;display:none}
        .thinking-overlay.open{display:block}
        .images-preview{display:flex;gap:8px;flex-wrap:wrap;padding:8px 0}
        .img-thumb{position:relative;width:60px;height:60px;border-radius:6px;overflow:hidden;border:1px solid var(--border)}
        .img-thumb img{width:100%;height:100%;object-fit:cover}
        .img-thumb .remove-img{position:absolute;top:2px;right:2px;background:rgba(0,0,0,.7);color:#fff;border:none;border-radius:50%;width:18px;height:18px;font-size:10px;cursor:pointer;display:flex;align-items:center;justify-content:center}
        .input-area{padding:16px 20px;border-top:1px solid var(--border);background:var(--bg2)}
        .input-row{display:flex;gap:8px;align-items:flex-end;max-width:800px;margin:0 auto}
        .input-wrapper{flex:1;position:relative}
        .input-wrapper textarea{width:100%;background:var(--bg3);color:var(--text);border:1px solid var(--border2);border-radius:var(--radius);padding:10px 14px;font-size:14px;font-family:inherit;resize:none;outline:none;min-height:44px;max-height:200px}
        .input-wrapper textarea:focus{border-color:var(--accent)}
        .input-wrapper textarea::placeholder{color:var(--text3)}
        .send-btn{background:var(--accent);color:#fff;border:none;border-radius:var(--radius);padding:10px 16px;cursor:pointer;font-size:14px;font-weight:500;white-space:nowrap}
        .send-btn:hover{background:var(--accent2)}
        .send-btn:disabled{opacity:.5;cursor:not-allowed}
        .stop-btn{background:var(--red);color:#fff;border:none;border-radius:var(--radius);padding:10px 16px;cursor:pointer;font-size:13px;font-weight:500}
        .attach-btn{background:var(--bg3);border:1px solid var(--border2);color:var(--text2);border-radius:var(--radius);padding:10px;cursor:pointer;font-size:14px}
        .attach-btn:hover{background:var(--bg4)}
        .status-bar{font-size:11px;color:var(--text3);padding:4px 0;text-align:center}
        .status-bar.streaming{color:var(--green)}
        .empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:var(--text3);gap:12px}
        .empty-state h2{font-size:20px;color:var(--text2)}
        .empty-state p{font-size:14px}
        .modal-overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:1000}
        .modal{background:var(--bg2);border:1px solid var(--border2);border-radius:12px;padding:24px;width:90%;max-width:500px}
        .modal h3{font-size:16px;margin-bottom:12px}
        .modal textarea,.modal input[type="text"],.modal input[type="password"],.modal input[type="number"]{width:100%;background:var(--bg3);color:var(--text);border:1px solid var(--border2);border-radius:6px;padding:10px;font-size:13px;font-family:inherit;outline:none}
        .modal textarea{resize:vertical;min-height:100px}
        .modal textarea:focus,.modal input:focus{border-color:var(--accent)}
        .modal-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:16px}
        .modal-actions button{padding:8px 16px;border-radius:6px;cursor:pointer;font-size:13px;border:1px solid var(--border2)}
        .modal-actions .primary{background:var(--accent);color:#fff;border-color:var(--accent)}
        .modal-actions .secondary{background:var(--bg3);color:var(--text)}
        .perf-info{font-size:11px;color:var(--text3);margin-top:4px;padding:6px 10px;background:var(--bg);border-radius:4px}
        .speed-indicator{font-size:11px;color:var(--green);display:inline;margin-left:8px}
        .error-msg{color:var(--red);font-size:13px;padding:12px 16px;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);border-radius:var(--radius)}
        .retry-btn{margin-top:8px;padding:6px 14px;background:var(--accent);color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:12px}
        .retry-btn:hover{background:var(--accent2)}
        .settings-panel{max-height:0;overflow:hidden;transition:max-height .3s ease;background:var(--bg2);border-bottom:1px solid var(--border)}
        .settings-panel.open{max-height:200px}
        .settings-inner{padding:12px 20px;display:flex;gap:16px;flex-wrap:wrap;max-width:800px;margin:0 auto}
        .setting-group{display:flex;flex-direction:column;gap:4px}
        .setting-group label{font-size:11px;color:var(--text3);font-weight:600;text-transform:uppercase}
        .setting-group input,.setting-group select{background:var(--bg3);color:var(--text);border:1px solid var(--border2);border-radius:4px;padding:4px 8px;font-size:12px;outline:none;width:80px}
        .setting-group input:focus,.setting-group select:focus{border-color:var(--accent)}
        .context-warning{font-size:11px;color:var(--yellow);padding:2px 8px;background:rgba(234,179,8,.1);border-radius:4px;margin-top:4px;text-align:center;max-width:800px;margin-left:auto;margin-right:auto;display:none}
        .context-warning.visible{display:block}
        .kbd{background:var(--bg3);border:1px solid var(--border2);border-radius:3px;padding:1px 5px;font-size:10px;font-family:monospace;color:var(--text3)}
        .shortcuts-modal .shortcut-row{display:flex;justify-content:space-between;padding:6px 0;font-size:13px}
        .shortcuts-modal .shortcut-row span:last-child{color:var(--text3)}
        .export-menu{position:absolute;bottom:100%;right:0;background:var(--bg2);border:1px solid var(--border2);border-radius:6px;padding:4px;min-width:120px;margin-bottom:4px;display:none;z-index:100}
        .export-menu.open{display:block}
        .export-menu button{display:block;width:100%;text-align:left;background:none;border:none;color:var(--text2);padding:6px 10px;font-size:12px;cursor:pointer;border-radius:4px}
        .export-menu button:hover{background:var(--bg3);color:var(--text)}
        @media(max-width:768px){
        .sidebar{position:fixed;left:-260px;z-index:100;transition:left .3s;height:100vh}
        .sidebar.open{left:0}
        .mobile-toggle{display:block!important}
        .settings-inner{gap:10px}
        }
        .mobile-toggle{display:none;position:fixed;top:12px;left:12px;z-index:99;background:var(--bg3);border:1px solid var(--border2);color:var(--text);padding:6px 10px;border-radius:6px;cursor:pointer;font-size:14px}
        .img-modal{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.9);z-index:2000;display:flex;align-items:center;justify-content:center;cursor:pointer}
        .img-modal img{max-width:90%;max-height:90%;border-radius:8px}
        </style>
        </head>
        <body>
        <button class="mobile-toggle" onclick="document.querySelector('.sidebar').classList.toggle('open')">&#9776;</button>
        <div class="sidebar">
        <div class="sidebar-header"><h2>NovaMLX Chat</h2><a href="/admin/dashboard" target="_blank">Dashboard</a></div>
        <button class="new-chat-btn" onclick="newChat()">+ New Chat <span class="kbd" style="float:right">Ctrl+N</span></button>
        <input class="search-box" id="chatSearch" placeholder="Search chats..." oninput="filterChats(this.value)">
        <div class="chat-list" id="chatList"></div>
        <div class="sidebar-bottom">
        <button class="topbar-btn" onclick="showApiKeyModal()">API Key</button>
        <div style="position:relative">
        <button class="topbar-btn" onclick="toggleExportMenu()">Export Chat</button>
        <div class="export-menu" id="exportMenu">
        <button onclick="exportChat('md')">Markdown</button>
        <button onclick="exportChat('json')">JSON</button>
        </div>
        </div>
        </div>
        </div>
        <div class="main">
        <div class="topbar">
        <h1>Nova<span>MLX</span></h1>
        <select id="modelSelect" onchange="saveCurrentChat()"></select>
        <button class="topbar-btn" id="settingsToggle" onclick="toggleSettings()" title="Model settings">&#9881;</button>
        <button class="topbar-btn" onclick="showSystemPrompt()" title="System prompt">&#9998;</button>
        <button class="topbar-btn" onclick="showShortcuts()" title="Keyboard shortcuts">?</button>
        </div>
        <div class="settings-panel" id="settingsPanel">
        <div class="settings-inner">
        <div class="setting-group"><label>Temperature</label><input type="number" id="cfgTemp" min="0" max="2" step="0.05" value="0.7" onchange="saveSettings()"></div>
        <div class="setting-group"><label>Max Tokens</label><input type="number" id="cfgMaxTokens" min="64" max="65536" step="64" value="4096" onchange="saveSettings()"></div>
        <div class="setting-group"><label>Top P</label><input type="number" id="cfgTopP" min="0" max="1" step="0.05" value="0.9" onchange="saveSettings()"></div>
        <div class="setting-group"><label>Top K</label><input type="number" id="cfgTopK" min="0" max="200" step="1" value="0" onchange="saveSettings()"></div>
        <div class="setting-group"><label>Min P</label><input type="number" id="cfgMinP" min="0" max="1" step="0.05" value="0" onchange="saveSettings()"></div>
        <div class="setting-group"><label>Repeat Penalty</label><input type="number" id="cfgRepeatPenalty" min="1" max="2" step="0.05" value="1.0" onchange="saveSettings()"></div>
        <div class="setting-group"><label>Context Limit</label><input type="number" id="cfgContextLimit" min="512" max="131072" step="1024" value="32768" onchange="saveSettings()"></div>
        <div class="setting-group" style="justify-content:flex-end"><button class="topbar-btn" onclick="resetSettingsToDefault()">Reset to Default</button></div>
        </div>
        </div>
        <div class="context-warning" id="contextWarning">Approaching context limit — older messages will be trimmed</div>
        <div class="messages-wrapper">
        <div class="messages" id="messages"></div>
        <button class="scroll-bottom-btn" id="scrollBottomBtn" onclick="scrollToBottom()">&#8595;</button>
        </div>
        <div class="input-area">
        <div id="imagesPreview" class="images-preview" style="display:none"></div>
        <div class="input-row">
        <button class="attach-btn" onclick="document.getElementById('fileInput').click()" title="Attach image" id="attachBtn" style="display:none">&#128206;</button>
        <input type="file" id="fileInput" accept="image/*" multiple style="display:none" onchange="handleFiles(this.files)">
        <div class="input-wrapper">
        <textarea id="userInput" placeholder="Type a message... (Enter to send, Shift+Enter for newline)" rows="1" onkeydown="handleInputKey(event)" oninput="autoResize(this)"></textarea>
        </div>
        <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
        <div class="status-bar" id="statusBar">Ready</div>
        </div>
        </div>
        <div id="systemPromptModal" class="modal-overlay" style="display:none">
        <div class="modal">
        <h3>System Prompt</h3>
        <textarea id="systemPromptText" placeholder="Enter system prompt..."></textarea>
        <div class="modal-actions">
        <button class="secondary" onclick="hideSystemPrompt()">Cancel</button>
        <button class="secondary" onclick="clearSystemPrompt()">Clear</button>
        <button class="primary" onclick="saveSystemPrompt()">Save</button>
        </div>
        </div>
        </div>
        <div id="imgModal" class="img-modal" style="display:none" onclick="this.style.display='none'">
        <img id="imgModalSrc" src="">
        </div>
        <div class="thinking-overlay" id="thinkingOverlay" onclick="closeThinkingPanel()"></div>
        <div class="thinking-panel" id="thinkingPanel">
        <div class="thinking-panel-header"><h3>Thinking Process</h3><button class="thinking-panel-close" onclick="closeThinkingPanel()">&#10005;</button></div>
        <div class="thinking-panel-body" id="thinkingPanelBody"></div>
        </div>
        <div id="apiKeyModal" class="modal-overlay" style="display:none">
        <div class="modal">
        <h3>API Key</h3>
        <p style="font-size:13px;color:var(--text3);margin-bottom:12px">Enter your API key to authenticate with the NovaMLX server.</p>
        <input type="password" id="apiKeyInput" placeholder="sk-...">
        <div class="modal-actions">
        <button class="secondary" onclick="hideApiKeyModal()">Cancel</button>
        <button class="primary" onclick="saveApiKey()">Save</button>
        </div>
        </div>
        </div>
        <div id="shortcutsModal" class="modal-overlay" style="display:none">
        <div class="modal shortcuts-modal">
        <h3>Keyboard Shortcuts</h3>
        <div class="shortcut-row"><span>New chat</span><span><span class="kbd">Ctrl</span>+<span class="kbd">N</span></span></div>
        <div class="shortcut-row"><span>Search chats</span><span><span class="kbd">Ctrl</span>+<span class="kbd">K</span></span></div>
        <div class="shortcut-row"><span>Send message</span><span><span class="kbd">Enter</span></span></div>
        <div class="shortcut-row"><span>New line</span><span><span class="kbd">Shift</span>+<span class="kbd">Enter</span></span></div>
        <div class="shortcut-row"><span>Previous input</span><span><span class="kbd">&uarr;</span></span></div>
        <div class="shortcut-row"><span>Next input</span><span><span class="kbd">&darr;</span></span></div>
        <div class="shortcut-row"><span>Close modal</span><span><span class="kbd">Esc</span></span></div>
        <div class="modal-actions"><button class="primary" onclick="document.getElementById('shortcutsModal').style.display='none'">Got it</button></div>
        </div>
        </div>
        <script>
        const BASE='';
        let apiKey='';
        function apiH(){return apiKey?{'Authorization':'Bearer '+apiKey}:{}}

        let state={chatId:null,messages:[],chatHistory:[],isStreaming:false,abortCtrl:null,models:[],systemPrompt:'',images:[],thinkingContent:'',streamTokens:0,streamStartTime:0,inputHistory:[],inputHistoryIdx:null,savedDraft:'',settings:{temperature:0.7,max_tokens:4096,top_p:0.9,top_k:0,min_p:0,repetition_penalty:1.0,context_limit:32768}};

        const renderer=new marked.Renderer();
        renderer.code=function(obj){
            const code=typeof obj==='object'?obj.text:obj;
            const lang=typeof obj==='object'?obj.lang:'';
            let highlighted;
            try{highlighted=lang?hljs.highlight(code,{language:lang}).value:hljs.highlightAuto(code).value}catch(e){highlighted=escHtml(code)}
            const langLabel=lang?`<span class="lang">${escHtml(lang)}</span>`:'<span class="lang">code</span>';
            return `<pre><div class="code-header">${langLabel}<button class="copy-code-btn" onclick="copyCode(this)">Copy</button></div><code class="hljs">${highlighted}</code></pre>`;
        };
        marked.setOptions({renderer,breaks:true,gfm:true});

        function escHtml(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
        function copyCode(btn){
            const code=btn.closest('pre').querySelector('code');
            navigator.clipboard.writeText(code.textContent);
            btn.textContent='Copied!';setTimeout(()=>btn.textContent='Copy',1500);
        }

        async function init(){
            apiKey=localStorage.getItem('novamlx_api_key')||'';
            loadSettings();
            await loadModels();
            await loadChatHistory();
            if(state.chatHistory.length>0)await loadChat(state.chatHistory[0].id);
            setupDragDrop();setupPaste();setupScrollButton();setupShortcuts();
            document.addEventListener('keydown',e=>{if(e.key==='Escape')closeAllModals()});
        }

        async function loadModels(){
            try{
                const r=await fetch(BASE+'/v1/models',{headers:apiH()});
                if(r.status===401){showApiKeyModal();setStatus('API key required');return}
                const d=await r.json();
                state.models=(d.data||[]).filter(m=>!m.id.includes('embed')&&!m.id.includes('rerank'));
                const sel=document.getElementById('modelSelect');
                sel.innerHTML=state.models.map(m=>`<option value="${m.id}">${m.id}</option>`).join('');
                if(state.models.length===0)setStatus('No models loaded. Load a model via Admin API.');
                else setStatus('Ready - '+state.models.length+' model(s) available');
            }catch(e){
                setStatus('Failed to load models: '+e.message);
            }
        }

        async function loadChatHistory(){
            try{
                const r=await fetch(BASE+'/v1/chat/history');
                if(r.ok){state.chatHistory=await r.json()}else{state.chatHistory=[]}
            }catch(e){state.chatHistory=[]}
            renderChatList();
        }

        function renderChatList(filter){
            const el=document.getElementById('chatList');
            let list=state.chatHistory;
            if(filter){const q=filter.toLowerCase();list=list.filter(c=>(c.title||'').toLowerCase().includes(q))}
            el.innerHTML=list.map(c=>`
                <div class="chat-item ${c.id===state.chatId?'active':''}" onclick="loadChat('${c.id}')">
                    <span style="color:var(--text3)">&#9998;</span>
                    <span class="title">${escHtml(c.title||'New Chat')}</span>
                    <button class="delete-btn" onclick="event.stopPropagation();deleteChat('${c.id}')">&#10005;</button>
                </div>
            `).join('');
        }

        function filterChats(q){renderChatList(q)}

        function newChat(){
            state.chatId='chat_'+Date.now();
            state.messages=[];
            state.systemPrompt=localStorage.getItem('novamlx_system_prompt')||'';
            renderMessages();renderChatList();setStatus('Ready');
        }

        async function loadChat(id){
            state.chatId=id;
            try{
                const r=await fetch(BASE+'/v1/chat/history/'+id);
                if(r.ok){
                    const chat=await r.json();
                    state.messages=(chat.messages||[]).map(m=>({...m,ts:m.ts||Date.now()}));
                    if(chat.systemPrompt)state.systemPrompt=chat.systemPrompt;
                    if(chat.model){const s=document.getElementById('modelSelect');if(s)s.value=chat.model;}
                }else{
                    const summary=state.chatHistory.find(c=>c.id===id);
                    state.messages=summary?[{role:'system',content:'Could not load chat details',ts:Date.now()}]:[];
                }
            }catch(e){state.messages=[]}
            renderMessages();renderChatList();setStatus('Ready');
        }

        async function deleteChat(id){
            try{await fetch(BASE+'/v1/chat/history/'+id,{method:'DELETE'})}catch(e){}
            state.chatHistory=state.chatHistory.filter(c=>c.id!==id);
            if(state.chatId===id){
                if(state.chatHistory.length>0)await loadChat(state.chatHistory[0].id);
                else newChat();
            }
            renderChatList();
        }

        async function saveCurrentChat(){
            if(!state.chatId)return;
            const lastMsg=state.messages.filter(m=>m.role==='user').pop();
            const title=lastMsg?lastMsg.content.substring(0,50):'New Chat';
            const stripImages=msgs=>msgs.map(m=>{const{images,...rest}=m;return rest});
            const chat={id:state.chatId,title:title,messages:stripImages(state.messages),model:document.getElementById('modelSelect').value,systemPrompt:state.systemPrompt,ts:Date.now()};
            try{
                await fetch(BASE+'/v1/chat/history',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(chat)});
                const idx=state.chatHistory.findIndex(c=>c.id===state.chatId);
                const summary={id:chat.id,title:chat.title,model:chat.model,ts:chat.ts,messageCount:chat.messages.length};
                if(idx>=0)state.chatHistory[idx]=summary;else state.chatHistory.unshift(summary);
            }catch(e){}
            renderChatList();
        }

        function renderMessages(){
            const el=document.getElementById('messages');
            if(state.messages.length===0){
                el.innerHTML='<div class="empty-state"><h2>NovaMLX Chat</h2><p>Start a conversation by typing a message below.</p><p style="font-size:12px;color:var(--text3)"><span class="kbd">Ctrl+N</span> new chat &nbsp; <span class="kbd">Ctrl+K</span> search</p></div>';
                return;
            }
            el.innerHTML=state.messages.map((m,i)=>renderMsg(m,i)).join('');
            el.querySelectorAll('.msg-body img').forEach(img=>{
                img.onclick=()=>{document.getElementById('imgModalSrc').src=img.src;document.getElementById('imgModal').style.display='flex'};
            });
            scrollToBottom();
        }

        function renderMsg(m,idx){
            const roleClass=m.role==='user'?'user':m.role==='assistant'?'assistant':'system';
            let bodyHtml='';
            if(m.thinking){
                const tkCount=m.thinking.split(/\\s+/).filter(Boolean).length;
                bodyHtml+=`<div class="thought-summary" onclick="openThinkingPanel(${idx})"><span class="dot"></span> Thought for ${m.thinkingTime||'?'}s &middot; ~${tkCount} words &middot; click to view</div>`;
            }
            if(m.error){
                bodyHtml+=`<div class="error-msg">${escHtml(m.content||'Unknown error')}</div>`;
                if(m.canRetry)bodyHtml+=`<button class="retry-btn" onclick="retryMsg(${idx})">Retry</button>`;
            } else if(m.role==='assistant'){
                const content=m.content||'';
                bodyHtml+=content?renderMd(content):'<span class="typing-cursor"></span>';
            } else {
                bodyHtml+=escHtml(m.content||'');
            }
            if(m.images&&m.images.length){
                bodyHtml+='<div class="images-preview">'+m.images.map(u=>`<div class="img-thumb"><img src="${u}"></div>`).join('')+'</div>';
            }
            let perfHtml='';
            if(m.perf)perfHtml=`<div class="perf-info">${m.perf}</div>`;
            const actions=m.role==='assistant'&&!m.error?`<div class="msg-actions">
                <button onclick="copyMsg(${idx})">Copy</button>
                <button onclick="regenerateMsg(${idx})">Regenerate</button>
            </div>`:(m.role==='user'?`<div class="msg-actions"><button onclick="editMsg(${idx})">Edit</button></div>`:'');
            const ts=m.ts?relativeTime(m.ts):'';
            return `<div class="msg" id="msg-${idx}">
                <div class="msg-header"><span class="role ${roleClass}">${m.role}</span><span class="ts" title="${m.ts?new Date(m.ts).toLocaleString():''}">${ts}</span></div>
                <div class="msg-body ${m.role==='user'?'user-msg':''}">${bodyHtml}</div>
                ${perfHtml}${actions}
            </div>`;
        }

        function renderMd(text){
            if(!text)return '';
            try{return marked.parse(cleanModelOutput(text))}catch(e){return escHtml(text)}
        }

        function cleanModelOutput(text){
            // Strip Qwen3.6 <|turn|> separators and partial residuals
            text=text.split('<'+'|turn|>').join('').split('<'+'|turn>').join('');
            // Strip legacy Qwen <|im_start|> / <|im_end|> markers
            text=text.split('<'+'|im_start|>').join('').split('<'+'|im_end|>').join('');
            // Strip <|endoftext|>
            text=text.split('<'+'|endoftext|>').join('');
            // Strip orphaned partial <| at end of stream
            text=text.replace(/<[|][a-z]*$/, '');
            return text.trimEnd();
        }

        function relativeTime(ts){
            const diff=Math.floor((Date.now()-ts)/1000);
            if(diff<60)return 'just now';
            if(diff<3600)return Math.floor(diff/60)+'m ago';
            if(diff<86400)return Math.floor(diff/3600)+'h ago';
            return new Date(ts).toLocaleDateString();
        }

        function scrollToBottom(){
            const el=document.getElementById('messages');
            el.scrollTop=el.scrollHeight;
        }

        function setupScrollButton(){
            const el=document.getElementById('messages');
            const btn=document.getElementById('scrollBottomBtn');
            el.addEventListener('scroll',()=>{
                const dist=el.scrollHeight-el.scrollTop-el.clientHeight;
                btn.classList.toggle('visible',dist>200);
            });
        }

        function setupShortcuts(){
            document.addEventListener('keydown',e=>{
                if(e.ctrlKey&&e.key==='n'){e.preventDefault();newChat()}
                if(e.ctrlKey&&e.key==='k'){e.preventDefault();document.getElementById('chatSearch').focus()}
            });
        }

        function handleInputKey(e){
            if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage()}
            // Up/Down arrow — navigate input history
            const ta=document.getElementById('userInput');
            if(e.key==='ArrowUp'&&ta.selectionStart===0&&ta.selectionEnd===0){
                e.preventDefault();
                if(state.inputHistory.length===0)return;
                if(state.inputHistoryIdx===null){state.savedDraft=ta.value;state.inputHistoryIdx=state.inputHistory.length}
                if(state.inputHistoryIdx>0){state.inputHistoryIdx--;ta.value=state.inputHistory[state.inputHistoryIdx];autoResize(ta)}
            }
            if(e.key==='ArrowDown'&&ta.selectionStart===ta.value.length&&ta.selectionEnd===ta.value.length){
                e.preventDefault();
                if(state.inputHistoryIdx===null)return;
                if(state.inputHistoryIdx>=state.inputHistory.length-1){state.inputHistoryIdx=null;ta.value=state.savedDraft;autoResize(ta)}
                else{state.inputHistoryIdx++;ta.value=state.inputHistory[state.inputHistoryIdx];autoResize(ta)}
            }
        }

        function autoResize(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,200)+'px'}
        function setStatus(s){document.getElementById('statusBar').textContent=s}

        function copyMsg(idx){navigator.clipboard.writeText(state.messages[idx].content||'');setStatus('Copied to clipboard')}

        async function regenerateMsg(idx){if(state.isStreaming)return;state.messages=state.messages.slice(0,idx);renderMessages();await streamResponse()}
        function editMsg(idx){const m=state.messages[idx];const nc=prompt('Edit message:',m.content);if(nc!==null&&nc.trim()){state.messages=state.messages.slice(0,idx);state.messages.push({role:'user',content:nc.trim(),ts:Date.now()});renderMessages();streamResponse()}}
        function retryMsg(idx){regenerateMsg(idx)}

        async function sendMessage(){
            const input=document.getElementById('userInput');
            const text=input.value.trim();
            if(!text&&state.images.length===0)return;
            if(state.isStreaming)return;
            const model=document.getElementById('modelSelect').value;
            if(!model){setStatus('No model selected');return}
            if(!state.chatId)newChat();
            // Push to input history
            if(text&&state.inputHistory[state.inputHistory.length-1]!==text){state.inputHistory.push(text)}
            state.inputHistoryIdx=null;state.savedDraft='';
            const userMsg={role:'user',content:text,images:state.images.length?[...state.images]:undefined,ts:Date.now()};
            state.messages.push(userMsg);
            state.images=[];
            document.getElementById('imagesPreview').style.display='none';
            document.getElementById('imagesPreview').innerHTML='';
            input.value='';autoResize(input);
            renderMessages();
            await streamResponse();
        }

        function buildApiMessages(){
            const limit=state.settings.context_limit||32768;
            const estTokens=msgs=>msgs.reduce((sum,m)=>sum+Math.ceil((m.content||'').length/3.5),0);
            let msgs=state.messages.slice(0,-1);
            const sysMsg=state.systemPrompt?[{role:'system',content:state.systemPrompt}]:[];
            let total=estTokens(sysMsg)+estTokens(msgs);
            while(total>limit*0.85&&msgs.length>1){total-=Math.ceil((msgs[0].content||'').length/3.5);msgs=msgs.slice(1)}
            const warn=document.getElementById('contextWarning');
            if(total>limit*0.7)warn.classList.add('visible');else warn.classList.remove('visible');
            const apiMsgs=[...sysMsg];
            msgs.forEach(m=>{
                const msg={role:m.role};
                if(m.images&&m.images.length){
                    msg.content=[{type:'text',text:m.content},...m.images.map(u=>({type:'image_url',image_url:{url:u}}))];
                }else{msg.content=m.content}
                apiMsgs.push(msg);
            });
            return apiMsgs;
        }

        async function streamResponse(){
            const model=document.getElementById('modelSelect').value;
            state.isStreaming=true;state.thinkingContent='';state.streamTokens=0;state.streamStartTime=0;
            state.abortCtrl=new AbortController();
            document.getElementById('sendBtn').style.display='none';
            const stopBtn=document.createElement('button');stopBtn.className='stop-btn';stopBtn.textContent='Stop';stopBtn.id='stopBtn';
            stopBtn.onclick=()=>state.abortCtrl.abort();
            document.querySelector('.input-row').appendChild(stopBtn);
            setStatus('Streaming...');

            const assistantIdx=state.messages.length;
            state.messages.push({role:'assistant',content:'',ts:Date.now()});
            renderMessages();
            const startTime=Date.now();

            // Show sync indicator immediately
            let syncTimerId=setInterval(()=>{
                const el=document.getElementById('msg-'+assistantIdx);
                if(!el)return;
                const bodyEl=el.querySelector('.msg-body');
                if(!bodyEl)return;
                const sec=((Date.now()-startTime)/1000).toFixed(0);
                bodyEl.innerHTML=`<div class="sync-indicator"><div class="sync-bar"></div><div class="sync-body"><div class="sync-dot"></div><div><div class="sync-text">Syncing with model...</div><div class="sync-timer">${sec}s elapsed</div></div></div></div>`;
                scrollToBottom();
            },250);
            // Fire first sync render immediately
            {const el=document.getElementById('msg-'+assistantIdx);if(el){const bodyEl=el.querySelector('.msg-body');if(bodyEl){bodyEl.innerHTML='<div class="sync-indicator"><div class="sync-bar"></div><div class="sync-body"><div class="sync-dot"></div><div><div class="sync-text">Syncing with model...</div><div class="sync-timer">0s elapsed</div></div></div></div>';}}}

            const apiMessages=buildApiMessages();
            const body={model,model:document.getElementById('modelSelect').value,messages:apiMessages,stream:true,stream_options:{include_usage:true},temperature:state.settings.temperature,max_tokens:state.settings.max_tokens,top_p:state.settings.top_p};
            if(state.settings.top_k>0)body.top_k=state.settings.top_k;
            if(state.settings.min_p>0)body.min_p=state.settings.min_p;
            if(state.settings.repetition_penalty&&state.settings.repetition_penalty!==1.0)body.repetition_penalty=state.settings.repetition_penalty;

            try{
                const resp=await fetch(BASE+'/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json',...apiH()},body:JSON.stringify(body),signal:state.abortCtrl.signal});
                if(resp.status===401){
                    state.messages.pop();state.isStreaming=false;state.abortCtrl=null;
                    document.getElementById('sendBtn').style.display='';const sb=document.getElementById('stopBtn');if(sb)sb.remove();
                    showApiKeyModal();setStatus('API key required');renderMessages();return;
                }
                if(!resp.ok){
                    let errMsg='Server error '+resp.status;
                    try{const ej=await resp.json();errMsg=ej.error?.message||errMsg}catch(e){}
                    throw new Error(errMsg);
                }
                const reader=resp.body.getReader();const decoder=new TextDecoder();
                let buffer='',thinkingStartTime=null,fullContent='',thinkingDone=false;
                let renderTimer=null,thinkingTokens=0,thinkingRefreshId=null;
                let firstDataReceived=false;

                function doRender(){
                    const el=document.getElementById('msg-'+assistantIdx);
                    if(!el)return;
                    let html='';
                    if(state.thinkingContent&&!thinkingDone){
                        const elapsed=((Date.now()-(thinkingStartTime||Date.now()))/1000).toFixed(1);
                        const speed=thinkingTokens>0?(thinkingTokens/parseFloat(elapsed)).toFixed(1):'--';
                        const ghost=escHtml(state.thinkingContent.slice(-80));
                        html+=`<div class="neural-pulse" onclick="openThinkingPanel(${assistantIdx})">
                            <div class="neural-pulse-bar"></div>
                            <div class="neural-pulse-body">
                                <div class="neural-pulse-dot"></div>
                                <div class="neural-pulse-metrics">
                                    <div class="neural-pulse-label">Thinking...</div>
                                    <div class="neural-pulse-stats">${elapsed}s &middot; ${thinkingTokens} tokens &middot; ${speed} tok/s</div>
                                </div>
                                <div class="neural-pulse-ghost">${ghost}</div>
                            </div>
                        </div>`;
                    }
                    html+=renderMd(fullContent);
                    const bodyEl=el.querySelector('.msg-body');
                    if(bodyEl){bodyEl.innerHTML=html;scrollToBottom();}
                }

                function scheduleRender(){
                    if(renderTimer)return;
                    renderTimer=setTimeout(()=>{renderTimer=null;doRender()},50);
                }

                function processBuffer(buf){
                    const lines=buf.split('\\n');
                    for(const line of lines){
                        if(line.startsWith(': '))continue;
                        if(!line.startsWith('data: '))continue;
                        const data=line.slice(6).trim();
                        if(data==='[DONE]')continue;
                        try{
                            const chunk=JSON.parse(data);
                            const delta=chunk.choices?.[0]?.delta;
                            if(delta){
                                if(!firstDataReceived&&(delta.reasoning_content||delta.content)){
                                    firstDataReceived=true;
                                    if(syncTimerId){clearInterval(syncTimerId);syncTimerId=null;}
                                }
                                if(delta.reasoning_content){
                                    const wasFirst=!thinkingStartTime;
                                    if(wasFirst)thinkingStartTime=Date.now();
                                    state.thinkingContent+=delta.reasoning_content;
                                    thinkingTokens++;
                                    if(wasFirst){
                                        doRender();
                                        thinkingRefreshId=setInterval(doRender,200);
                                    } else {
                                        scheduleRender();
                                    }
                                }
                                if(delta.content){
                                    if(!state.streamStartTime)state.streamStartTime=Date.now();
                                    if(!thinkingDone&&state.thinkingContent){thinkingDone=true;if(thinkingRefreshId){clearInterval(thinkingRefreshId);thinkingRefreshId=null;}}
                                    fullContent+=delta.content;
                                    state.streamTokens++;
                                    scheduleRender();
                                }
                                // Diagnostic: detect if thinking leaks into content
                                if(delta.content && thinkingDone===false && !delta.reasoning_content && !state.thinkingContent){
                                    // Content arrived before any thinking — OK for non-thinking models
                                }
                            }
                            if(chunk.usage){
                                const u=chunk.usage;
                                const elapsed=((Date.now()-startTime)/1000).toFixed(1);
                                const tps=u.completion_tokens>0?(u.completion_tokens/parseFloat(elapsed)).toFixed(1):'--';
                                state.messages[assistantIdx].perf=`Prompt: ${u.prompt_tokens} | Completion: ${u.completion_tokens} | ${tps} tokens/s | ${elapsed}s`;
                            }
                        }catch(e){console.warn('SSE parse error:',e)}
                    }
                }
                while(true){
                    const {done,value}=await reader.read();
                    if(done)break;
                    buffer+=decoder.decode(value,{stream:true});
                    const lines=buffer.split('\\n');buffer=lines.pop()||'';
                    processBuffer(lines.join('\\n'));
                }
                // Flush remaining buffer after stream ends
                if(buffer.trim())processBuffer(buffer);buffer='';

                if(renderTimer)clearTimeout(renderTimer);
                if(thinkingRefreshId){clearInterval(thinkingRefreshId);thinkingRefreshId=null;}
                if(syncTimerId){clearInterval(syncTimerId);syncTimerId=null;}

                // Diagnostic analysis
                (function(){
                    const tLen=state.thinkingContent.length;
                    const cLen=fullContent.length;
                    const thinkPreview=state.thinkingContent.substring(0,200);
                    const contentPreview=fullContent.substring(0,200);

                    console.group('[NovaMLX SSE Diagnostic]');
                    console.log('thinkingContent: '+tLen+' chars');
                    console.log('fullContent: '+cLen+' chars');
                    console.log('thinkingDone: '+thinkingDone);

                    // Detect: thinking content looks like actual answer
                    const thinkingHasSteps=/Step \\d/i.test(state.thinkingContent);
                    const thinkingHasAnswer=/^.*\\d+\\s*\\+\\s*\\d+\\s*=\\s*\\d+.*$/m.test(state.thinkingContent);
                    const contentHasStepByStep=/Step \\d|step-by-step/i.test(fullContent);
                    const contentLooksLikeThinking=/Analyze|Identify|Consider|Formulate|Draft/i.test(fullContent) && cLen>500;

                    if(thinkingHasSteps&&cLen<50){
                        console.warn('BUG DETECTED: thinking contains step-by-step answer but content is empty/short. Thinking is leaking.');
                        console.log('thinking (first 300):',thinkPreview);
                        console.log('content (first 300):',contentPreview);
                    }else if(contentLooksLikeThinking&&tLen>0){
                        console.warn('BUG DETECTED: content looks like thinking process (meta-reasoning). Content may contain thinking tokens.');
                        console.log('content (first 300):',contentPreview);
                        console.log('thinking (first 300):',thinkPreview);
                    }else if(tLen===0&&cLen===0){
                        console.error('BUG DETECTED: both thinking and content are empty!');
                    }else if(tLen>0&&cLen===0){
                        console.warn('BUG DETECTED: thinking exists but content is empty. All output went to thinking.');
                    }else{
                        console.log('OK: thinking and content appear correctly separated');
                    }

                    // Log raw SSE field names received
                    console.log('thinking preview: '+thinkPreview.substring(0,150));
                    console.log('content preview: '+contentPreview.substring(0,150));
                    console.groupEnd();
                })();

                state.messages[assistantIdx].content=fullContent;
                if(state.thinkingContent){
                    state.messages[assistantIdx].thinking=state.thinkingContent;
                    state.messages[assistantIdx].thinkingTime=((Date.now()-(thinkingStartTime||startTime))/1000).toFixed(1);
                }
                const elapsed=((Date.now()-startTime)/1000).toFixed(1);
                setStatus(`Completed in ${elapsed}s`);
                saveCurrentChat();
            }catch(e){
                if(e.name==='AbortError'){
                    setStatus('Stopped');
                }else{
                    setStatus('Error: '+e.message);
                    state.messages[assistantIdx].content=e.message;
                    state.messages[assistantIdx].error=true;
                    state.messages[assistantIdx].canRetry=true;
                }
            }finally{
                state.isStreaming=false;state.abortCtrl=null;
                document.getElementById('sendBtn').style.display='';
                const sb=document.getElementById('stopBtn');if(sb)sb.remove();
                renderMessages();
            }
        }

        function showSystemPrompt(){document.getElementById('systemPromptText').value=state.systemPrompt;document.getElementById('systemPromptModal').style.display='flex'}
        function hideSystemPrompt(){document.getElementById('systemPromptModal').style.display='none'}
        function clearSystemPrompt(){state.systemPrompt='';hideSystemPrompt();localStorage.removeItem('novamlx_system_prompt')}
        function saveSystemPrompt(){state.systemPrompt=document.getElementById('systemPromptText').value;localStorage.setItem('novamlx_system_prompt',state.systemPrompt);hideSystemPrompt()}

        function handleFiles(files){
            Array.from(files).forEach(f=>{
                if(f.size>10*1024*1024){setStatus('Image too large (max 10MB)');return}
                const reader=new FileReader();reader.onload=e=>{state.images.push(e.target.result);renderImagesPreview()};reader.readAsDataURL(f);
            });
        }
        function renderImagesPreview(){const el=document.getElementById('imagesPreview');if(state.images.length===0){el.style.display='none';return}el.style.display='flex';el.innerHTML=state.images.map((u,i)=>`<div class="img-thumb"><img src="${u}"><button class="remove-img" onclick="removeImage(${i})">&#10005;</button></div>`).join('')}
        function removeImage(i){state.images.splice(i,1);renderImagesPreview()}
        function setupDragDrop(){const b=document.body;b.addEventListener('dragover',e=>{e.preventDefault()});b.addEventListener('drop',e=>{e.preventDefault();if(e.dataTransfer.files.length)handleFiles(e.dataTransfer.files)})}
        function setupPaste(){document.addEventListener('paste',e=>{const items=e.clipboardData?.items;if(!items)return;for(const item of items){if(item.type.startsWith('image/')){e.preventDefault();handleFiles([item.getAsFile()]);break}}})}

        function showApiKeyModal(){document.getElementById('apiKeyModal').style.display='flex'}
        function hideApiKeyModal(){document.getElementById('apiKeyModal').style.display='none'}
        function saveApiKey(){apiKey=document.getElementById('apiKeyInput').value.trim();localStorage.setItem('novamlx_api_key',apiKey);hideApiKeyModal();loadModels()}
        function showShortcuts(){document.getElementById('shortcutsModal').style.display='flex'}
        function closeAllModals(){['systemPromptModal','apiKeyModal','shortcutsModal'].forEach(id=>{document.getElementById(id).style.display='none'});closeThinkingPanel()}

        function openThinkingPanel(idx){
            const m=state.messages[idx];
            const thinking=m?.thinking||state.thinkingContent;
            if(!thinking)return;
            const body=document.getElementById('thinkingPanelBody');
            body.innerHTML=renderMd(thinking);
            document.getElementById('thinkingPanel').classList.add('open');
            document.getElementById('thinkingOverlay').classList.add('open');
        }
        function closeThinkingPanel(){
            document.getElementById('thinkingPanel').classList.remove('open');
            document.getElementById('thinkingOverlay').classList.remove('open');
        }

        function toggleSettings(){document.getElementById('settingsPanel').classList.toggle('open')}
        function saveSettings(){
            state.settings={
                temperature:parseFloat(document.getElementById('cfgTemp').value)||0.7,
                max_tokens:parseInt(document.getElementById('cfgMaxTokens').value)||4096,
                top_p:parseFloat(document.getElementById('cfgTopP').value)||0.9,
                top_k:parseInt(document.getElementById('cfgTopK').value)||0,
                min_p:parseFloat(document.getElementById('cfgMinP').value)||0,
                repetition_penalty:parseFloat(document.getElementById('cfgRepeatPenalty').value)||1.0,
                context_limit:parseInt(document.getElementById('cfgContextLimit').value)||32768
            };
            localStorage.setItem('novamlx_settings',JSON.stringify(state.settings));
        }
        function loadSettings(){
            const saved=JSON.parse(localStorage.getItem('novamlx_settings')||'null');
            if(saved)state.settings={...state.settings,...saved};
            syncSettingsUI();
        }
        function syncSettingsUI(){
            document.getElementById('cfgTemp').value=state.settings.temperature;
            document.getElementById('cfgMaxTokens').value=state.settings.max_tokens;
            document.getElementById('cfgTopP').value=state.settings.top_p;
            document.getElementById('cfgTopK').value=state.settings.top_k;
            document.getElementById('cfgMinP').value=state.settings.min_p||0;
            document.getElementById('cfgRepeatPenalty').value=state.settings.repetition_penalty||1.0;
            document.getElementById('cfgContextLimit').value=state.settings.context_limit;
        }
        function resetSettingsToDefault(){
            const defaults={temperature:0.7,max_tokens:4096,top_p:0.9,top_k:0,min_p:0,repetition_penalty:1.0,context_limit:32768};
            state.settings={...defaults};
            localStorage.removeItem('novamlx_settings');
            syncSettingsUI();
            setStatus('Settings reset to defaults');
        }

        function toggleExportMenu(){document.getElementById('exportMenu').classList.toggle('open')}
        document.addEventListener('click',e=>{if(!e.target.closest('.export-menu')&&!e.target.closest('[onclick*="toggleExportMenu"]'))document.getElementById('exportMenu').classList.remove('open')});

        function exportChat(fmt){
            document.getElementById('exportMenu').classList.remove('open');
            if(!state.messages.length){setStatus('Nothing to export');return}
            let content,name;
            if(fmt==='json'){
                content=JSON.stringify({id:state.chatId,messages:state.messages,model:document.getElementById('modelSelect').value,systemPrompt:state.systemPrompt,exportedAt:new Date().toISOString()},null,2);
                name='chat-'+state.chatId+'.json';
            }else{
                content=(state.systemPrompt?'# System Prompt\\n\\n'+state.systemPrompt+'\\n\\n---\\n\\n':'')+
                    state.messages.map(m=>'## '+m.role.charAt(0).toUpperCase()+m.role.slice(1)+'\\n\\n'+(m.content||'')).join('\\n\\n---\\n\\n');
                name='chat-'+state.chatId+'.md';
            }
            const blob=new Blob([content],{type:fmt==='json'?'application/json':'text/markdown'});
            const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();URL.revokeObjectURL(a.href);
            setStatus('Exported as '+fmt.toUpperCase());
        }

        init();
        </script>
        </body>
        </html>
        """
    }
}

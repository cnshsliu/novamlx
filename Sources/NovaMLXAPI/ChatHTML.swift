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
        .chat-list{flex:1;overflow-y:auto;padding:4px 8px}
        .chat-item{padding:10px 12px;border-radius:6px;cursor:pointer;font-size:13px;color:var(--text2);display:flex;align-items:center;gap:8px;margin-bottom:2px;position:relative}
        .chat-item:hover{background:var(--bg3);color:var(--text)}
        .chat-item.active{background:var(--bg3);color:var(--text)}
        .chat-item .delete-btn{position:absolute;right:8px;opacity:0;background:none;border:none;color:var(--red);cursor:pointer;font-size:12px;padding:2px 4px}
        .chat-item:hover .delete-btn{opacity:1}
        .chat-item .title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
        .main{flex:1;display:flex;flex-direction:column;min-width:0}
        .topbar{padding:12px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;background:var(--bg2)}
        .topbar h1{font-size:16px;font-weight:600;flex:1}
        .topbar h1 span{color:var(--accent)}
        select{background:var(--bg3);color:var(--text);border:1px solid var(--border2);border-radius:6px;padding:6px 10px;font-size:13px;cursor:pointer;outline:none}
        select:focus{border-color:var(--accent)}
        .topbar-btn{background:var(--bg3);border:1px solid var(--border2);color:var(--text2);padding:6px 10px;border-radius:6px;cursor:pointer;font-size:12px}
        .topbar-btn:hover{background:var(--bg4);color:var(--text)}
        .messages{flex:1;overflow-y:auto;padding:20px;scroll-behavior:smooth}
        .msg{margin-bottom:16px;max-width:800px;margin-left:auto;margin-right:auto}
        .msg-header{font-size:11px;color:var(--text3);margin-bottom:4px;display:flex;align-items:center;gap:8px}
        .msg-header .role{font-weight:600;text-transform:uppercase}
        .msg-header .role.user{color:var(--blue)}
        .msg-header .role.assistant{color:var(--green)}
        .msg-header .role.system{color:var(--yellow)}
        .msg-body{font-size:14px;line-height:1.7;color:var(--text);padding:12px 16px;border-radius:var(--radius);background:var(--bg2);border:1px solid var(--border)}
        .msg-body.user-msg{background:var(--bg3)}
        .msg-body p{margin-bottom:8px}
        .msg-body p:last-child{margin-bottom:0}
        .msg-body code{background:var(--code-bg);padding:2px 6px;border-radius:4px;font-size:13px;font-family:"SF Mono",Monaco,"Cascadia Code",monospace}
        .msg-body pre{background:var(--code-bg);border-radius:6px;padding:12px 16px;overflow-x:auto;margin:8px 0;position:relative}
        .msg-body pre code{background:none;padding:0;font-size:13px}
        .msg-body ul,.msg-body ol{margin:8px 0;padding-left:24px}
        .msg-body li{margin-bottom:4px}
        .msg-body blockquote{border-left:3px solid var(--accent);padding-left:12px;color:var(--text2);margin:8px 0}
        .msg-body a{color:var(--accent)}
        .msg-body h1,.msg-body h2,.msg-body h3{margin:12px 0 8px;color:var(--text)}
        .msg-body h1{font-size:20px}.msg-body h2{font-size:17px}.msg-body h3{font-size:15px}
        .msg-body table{border-collapse:collapse;width:100%;margin:8px 0}
        .msg-body th,.msg-body td{border:1px solid var(--border2);padding:6px 10px;text-align:left;font-size:13px}
        .msg-body th{background:var(--bg3)}
        .copy-code-btn{position:absolute;top:6px;right:6px;background:var(--bg4);border:1px solid var(--border2);color:var(--text3);padding:3px 8px;border-radius:4px;font-size:11px;cursor:pointer;opacity:0;transition:opacity .2s}
        pre:hover .copy-code-btn{opacity:1}
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
        .modal textarea{width:100%;background:var(--bg3);color:var(--text);border:1px solid var(--border2);border-radius:6px;padding:10px;font-size:13px;font-family:inherit;resize:vertical;min-height:100px;outline:none}
        .modal textarea:focus{border-color:var(--accent)}
        .modal-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:16px}
        .modal-actions button{padding:8px 16px;border-radius:6px;cursor:pointer;font-size:13px;border:1px solid var(--border2)}
        .modal-actions .primary{background:var(--accent);color:#fff;border-color:var(--accent)}
        .modal-actions .secondary{background:var(--bg3);color:var(--text)}
        .perf-info{font-size:11px;color:var(--text3);margin-top:4px;padding:6px 10px;background:var(--bg);border-radius:4px;display:none}
        .perf-info.open{display:block}
        .token-counter{font-size:11px;color:var(--text3);text-align:right;padding-top:4px}
        @media(max-width:768px){
        .sidebar{position:fixed;left:-260px;z-index:100;transition:left .3s;height:100vh}
        .sidebar.open{left:0}
        .mobile-toggle{display:block!important}
        }
        .mobile-toggle{display:none;position:fixed;top:12px;left:12px;z-index:99;background:var(--bg3);border:1px solid var(--border2);color:var(--text);padding:6px 10px;border-radius:6px;cursor:pointer;font-size:14px}
        .msg-body img{max-width:100%;border-radius:6px;cursor:pointer;margin:4px 0}
        .img-modal{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.9);z-index:2000;display:flex;align-items:center;justify-content:center;cursor:pointer}
        .img-modal img{max-width:90%;max-height:90%;border-radius:8px}
        </style>
        </head>
        <body>
        <button class="mobile-toggle" onclick="document.querySelector('.sidebar').classList.toggle('open')">&#9776;</button>
        <div class="sidebar">
        <div class="sidebar-header"><h2>NovaMLX Chat</h2><a href="/admin/dashboard" target="_blank">Dashboard</a></div>
        <button class="new-chat-btn" onclick="newChat()">+ New Chat</button>
        <div class="chat-list" id="chatList"></div>
        <div style="padding:12px 16px;border-top:1px solid var(--border)">
        <button class="topbar-btn" style="width:100%" onclick="showApiKeyModal()">API Key</button>
        </div>
        </div>
        <div class="main">
        <div class="topbar">
        <h1>Nova<span>MLX</span></h1>
        <select id="modelSelect" onchange="saveCurrentChat()"></select>
        <button class="topbar-btn" onclick="showSystemPrompt()" title="System prompt">&#9881;</button>
        </div>
        <div class="messages" id="messages"></div>
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
        <div id="apiKeyModal" class="modal-overlay" style="display:none">
        <div class="modal">
        <h3>API Key</h3>
        <p style="font-size:13px;color:var(--text3);margin-bottom:12px">Enter your API key to authenticate with the NovaMLX server.</p>
        <input type="password" id="apiKeyInput" placeholder="sk-..." style="width:100%;background:var(--bg3);color:var(--text);border:1px solid var(--border2);border-radius:6px;padding:10px;font-size:13px;outline:none">
        <div class="modal-actions">
        <button class="secondary" onclick="hideApiKeyModal()">Cancel</button>
        <button class="primary" onclick="saveApiKey()">Save</button>
        </div>
        </div>
        </div>
        <script>
        const BASE='';
        let apiKey='';
        function apiH(){return apiKey?{'Authorization':'Bearer '+apiKey}:{}}

        let state={chatId:null,messages:[],chatHistory:[],isStreaming:false,abortCtrl:null,models:[],modelTypes:{},systemPrompt:'',images:[],isDragOver:false,thinkingContent:''};

        async function init(){
            apiKey=localStorage.getItem('novamlx_api_key')||'';
            await loadModels();
            loadChatHistory();
            if(state.chatHistory.length>0)loadChat(state.chatHistory[0].id);
            setupDragDrop();
            setupPaste();
            document.addEventListener('keydown',e=>{if(e.key==='Escape')hideSystemPrompt()});
        }

        async function loadModels(){
            try{
                const r=await fetch(BASE+'/v1/models',{headers:apiH()});
                const d=await r.json();
                state.models=(d.data||[]).filter(m=>!m.id.includes('embed')&&!m.id.includes('rerank'));
                const sel=document.getElementById('modelSelect');
                sel.innerHTML=state.models.map(m=>`<option value="${m.id}">${m.id}</option>`).join('');
                try{
                    const sr=await fetch(BASE+'/v1/models/status',{headers:apiH()});
                    const sd=await sr.json();
                    if(sd.models)state.modelTypes=sd.models;
                }catch(e){}
                const hasVLM=state.models.some(m=>{const t=state.modelTypes[m.id];return t&&(t.type==='vlm'||t.is_vlm)});
                document.getElementById('attachBtn').style.display=hasVLM?'block':'none';
                if(!apiKey&&state.models.length===0)setStatus('No models loaded. Load a model via Admin API.');
            }catch(e){
                if(e.message&&e.message.includes('401'))showApiKeyModal();
                setStatus('Failed to load models: '+e.message);
            }
        }

        function loadChatHistory(){
            state.chatHistory=JSON.parse(localStorage.getItem('novamlx_chat_history')||'[]');
            renderChatList();
        }

        function renderChatList(){
            const el=document.getElementById('chatList');
            el.innerHTML=state.chatHistory.map(c=>`
                <div class="chat-item ${c.id===state.chatId?'active':''}" onclick="loadChat('${c.id}')">
                    <span style="color:var(--text3)">&#9998;</span>
                    <span class="title">${escHtml(c.title||'New Chat')}</span>
                    <button class="delete-btn" onclick="event.stopPropagation();deleteChat('${c.id}')">&#10005;</button>
                </div>
            `).join('');
        }

        function newChat(){
            state.chatId='chat_'+Date.now();
            state.messages=[];
            state.systemPrompt=localStorage.getItem('novamlx_system_prompt')||'';
            renderMessages();
            renderChatList();
            setStatus('Ready');
        }

        function loadChat(id){
            state.chatId=id;
            const chat=state.chatHistory.find(c=>c.id===id);
            if(chat){
                state.messages=chat.messages||[];
                if(chat.systemPrompt)state.systemPrompt=chat.systemPrompt;
                if(chat.model){const s=document.getElementById('modelSelect');if(s)s.value=chat.model;}
            }
            renderMessages();
            renderChatList();
            setStatus('Ready');
        }

        function deleteChat(id){
            state.chatHistory=state.chatHistory.filter(c=>c.id!==id);
            localStorage.setItem('novamlx_chat_history',JSON.stringify(state.chatHistory));
            if(state.chatId===id){
                if(state.chatHistory.length>0)loadChat(state.chatHistory[0].id);
                else newChat();
            }
            renderChatList();
        }

        function saveCurrentChat(){
            if(!state.chatId)return;
            const lastMsg=state.messages.filter(m=>m.role==='user').pop();
            const title=lastMsg?lastMsg.content.substring(0,50):'New Chat';
            const stripImages=msgs=>msgs.map(m=>({...m,images:undefined}));
            const chat={id:state.chatId,title:title,messages:stripImages(state.messages),model:document.getElementById('modelSelect').value,systemPrompt:state.systemPrompt,ts:Date.now()};
            const idx=state.chatHistory.findIndex(c=>c.id===state.chatId);
            if(idx>=0)state.chatHistory[idx]=chat;else state.chatHistory.unshift(chat);
            if(state.chatHistory.length>50)state.chatHistory=state.chatHistory.slice(0,50);
            localStorage.setItem('novamlx_chat_history',JSON.stringify(state.chatHistory));
            renderChatList();
        }

        function renderMessages(){
            const el=document.getElementById('messages');
            if(state.messages.length===0){
                el.innerHTML='<div class="empty-state"><h2>NovaMLX Chat</h2><p>Start a conversation by typing a message below.</p></div>';
                return;
            }
            el.innerHTML=state.messages.map((m,i)=>renderMsg(m,i)).join('');
            el.querySelectorAll('pre').forEach(pre=>{
                if(!pre.querySelector('.copy-code-btn')){
                    const btn=document.createElement('button');
                    btn.className='copy-code-btn';
                    btn.textContent='Copy';
                    btn.onclick=()=>{navigator.clipboard.writeText(pre.textContent.replace('Copy','').trim());btn.textContent='Copied!';setTimeout(()=>btn.textContent='Copy',1500)};
                    pre.style.position='relative';
                    pre.appendChild(btn);
                }
            });
            el.querySelectorAll('.msg-body img').forEach(img=>{
                img.onclick=()=>{document.getElementById('imgModalSrc').src=img.src;document.getElementById('imgModal').style.display='flex'};
            });
            scrollToBottom();
        }

        function renderMsg(m,idx){
            const roleClass=m.role==='user'?'user':m.role==='assistant'?'assistant':'system';
            let bodyHtml='';
            if(m.thinking){
                bodyHtml+=`<div class="thinking-bubble">
                    <div class="thinking-toggle" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">
                        <span class="arrow">&#9654;</span> Thinking${m.thinkingTime?' ('+m.thinkingTime+'s)':''}
                    </div>
                    <div class="thinking-content">${renderMd(m.thinking)}</div>
                </div>`;
            }
            if(m.role==='assistant'){
                bodyHtml+=renderMd(m.content||'');
            } else {
                bodyHtml+=escHtml(m.content||'');
            }
            if(m.images&&m.images.length){
                bodyHtml+='<div class="images-preview">'+m.images.map(u=>`<div class="img-thumb"><img src="${u}"></div>`).join('')+'</div>';
            }
            let perfHtml='';
            if(m.perf){
                perfHtml=`<div class="perf-info">${m.perf}</div>`;
            }
            const actions=m.role==='assistant'?`<div class="msg-actions">
                <button onclick="copyMsg(${idx})">Copy</button>
                <button onclick="regenerateMsg(${idx})">Regenerate</button>
            </div>`:`<div class="msg-actions">
                <button onclick="editMsg(${idx})">Edit</button>
            </div>`;
            return `<div class="msg" id="msg-${idx}">
                <div class="msg-header"><span class="role ${roleClass}">${m.role}</span>${m.ts?new Date(m.ts).toLocaleTimeString():''}</div>
                <div class="msg-body ${m.role==='user'?'user-msg':''}">${bodyHtml}</div>
                ${perfHtml}${actions}
            </div>`;
        }

        function renderMd(text){
            if(!text)return '';
            let html=escHtml(text);
            html=html.replace(/```(\\w*)\\n([\\s\\S]*?)```/g,(_,lang,code)=>`<pre><code>${code.trim()}</code></pre>`);
            html=html.replace(/`([^`]+)`/g,'<code>$1</code>');
            html=html.replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>');
            html=html.replace(/\\*(.+?)\\*/g,'<em>$1</em>');
            html=html.replace(/^### (.+)$/gm,'<h3>$1</h3>');
            html=html.replace(/^## (.+)$/gm,'<h2>$1</h2>');
            html=html.replace(/^# (.+)$/gm,'<h1>$1</h1>');
            html=html.replace(/^> (.+)$/gm,'<blockquote>$1</blockquote>');
            html=html.replace(/^[-*] (.+)$/gm,'<li>$1</li>');
            html=html.replace(/(<li>.*<\\/li>)/gs,match=>`<ul>${match}</ul>`);
            html=html.replace(/\\n/g,'<br>');
            html=html.replace(/<br><br>/g,'</p><p>');
            html='<p>'+html+'</p>';
            html=html.replace(/<p><\\/p>/g,'');
            html=html.replace(/<p>(<h[123]>)/g,'$1');
            html=html.replace(/(<\\/h[123]>)<\\/p>/g,'$1');
            html=html.replace(/<p>(<pre>)/g,'$1');
            html=html.replace(/(<\\/pre>)<\\/p>/g,'$1');
            html=html.replace(/<p>(<ul>)/g,'$1');
            html=html.replace(/(<\\/ul>)<\\/p>/g,'$1');
            html=html.replace(/<p>(<blockquote>)/g,'$1');
            html=html.replace(/(<\\/blockquote>)<\\/p>/g,'$1');
            return html;
        }

        function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}

        function scrollToBottom(){
            const el=document.getElementById('messages');
            el.scrollTop=el.scrollHeight;
        }

        async function sendMessage(){
            const input=document.getElementById('userInput');
            const text=input.value.trim();
            if(!text&&state.images.length===0)return;
            if(state.isStreaming)return;
            const model=document.getElementById('modelSelect').value;
            if(!model){setStatus('No model selected');return}

            const userMsg={role:'user',content:text,images:state.images.length?[...state.images]:undefined,ts:Date.now()};
            state.messages.push(userMsg);
            state.images=[];
            document.getElementById('imagesPreview').style.display='none';
            document.getElementById('imagesPreview').innerHTML='';
            input.value='';
            autoResize(input);
            renderMessages();

            await streamResponse();
        }

        async function streamResponse(){
            const model=document.getElementById('modelSelect').value;
            state.isStreaming=true;
            state.thinkingContent='';
            state.abortCtrl=new AbortController();
            document.getElementById('sendBtn').style.display='none';
            const stopBtn=document.createElement('button');
            stopBtn.className='stop-btn';
            stopBtn.textContent='Stop';
            stopBtn.id='stopBtn';
            stopBtn.onclick=()=>state.abortCtrl.abort();
            document.querySelector('.input-row').appendChild(stopBtn);
            setStatus('Streaming...');

            const assistantIdx=state.messages.length;
            state.messages.push({role:'assistant',content:'',ts:Date.now()});
            const startTime=Date.now();

            const apiMessages=[];
            if(state.systemPrompt)apiMessages.push({role:'system',content:state.systemPrompt});
            state.messages.slice(0,-1).forEach(m=>{
                const msg={role:m.role};
                if(m.images&&m.images.length){
                    msg.content=[{type:'text',text:m.content},...m.images.map(u=>({type:'image_url',image_url:{url:u}}))];
                } else {
                    msg.content=m.content;
                }
                apiMessages.push(msg);
            });

            const body={model:model,messages:apiMessages,stream:true,stream_options:{include_usage:true}};

            try{
                const resp=await fetch(BASE+'/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json',...apiH()},body:JSON.stringify(body),signal:state.abortCtrl.signal});
                const reader=resp.body.getReader();
                const decoder=new TextDecoder();
                let buffer='';
                let thinkingStartTime=null;
                let fullContent='';
                let thinkingDone=false;

                while(true){
                    const {done,value}=await reader.read();
                    if(done)break;
                    buffer+=decoder.decode(value,{stream:true});
                    const lines=buffer.split('\\n');
                    buffer=lines.pop()||'';

                    for(const line of lines){
                        if(line.startsWith(': '))continue;
                        if(!line.startsWith('data: '))continue;
                        const data=line.slice(6).trim();
                        if(data==='[DONE]')continue;
                        try{
                            const chunk=JSON.parse(data);
                            const delta=chunk.choices?.[0]?.delta;
                            if(delta){
                                if(delta.reasoning_content){
                                    if(!thinkingStartTime)thinkingStartTime=Date.now();
                                    state.thinkingContent+=delta.reasoning_content;
                                    updateStreamingMessage(assistantIdx,fullContent,state.thinkingContent,thinkingDone);
                                }
                                if(delta.content){
                                    if(!thinkingDone&&state.thinkingContent){thinkingDone=true}
                                    fullContent+=delta.content;
                                    updateStreamingMessage(assistantIdx,fullContent,state.thinkingContent,thinkingDone);
                                }
                            }
                            if(chunk.usage){
                                const u=chunk.usage;
                                state.messages[assistantIdx].perf=`Prompt: ${u.prompt_tokens} tokens | Completion: ${u.completion_tokens} tokens`;
                            }
                        }catch(e){}
                    }
                }

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
                } else {
                    setStatus('Error: '+e.message);
                    state.messages[assistantIdx].content='Error: '+e.message;
                }
            } finally {
                state.isStreaming=false;
                state.abortCtrl=null;
                document.getElementById('sendBtn').style.display='';
                const sb=document.getElementById('stopBtn');if(sb)sb.remove();
                renderMessages();
            }
        }

        function updateStreamingMessage(idx,content,thinking,thinkingDone){
            const el=document.getElementById('msg-'+idx);
            if(!el)return;
            let html='';
            if(thinking&&!thinkingDone){
                html+=`<div class="thinking-bubble"><div class="thinking-toggle open" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">
                    <span class="arrow" style="transform:rotate(90deg)">&#9654;</span> Thinking...
                </div><div class="thinking-content open">${renderMd(thinking)}</div></div>`;
            } else if(thinking&&thinkingDone){
                html+=`<div class="thinking-bubble"><div class="thinking-toggle" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">
                    <span class="arrow">&#9654;</span> Thinking
                </div><div class="thinking-content">${renderMd(thinking)}</div></div>`;
            }
            html+=renderMd(content);
            const bodyEl=el.querySelector('.msg-body');
            if(bodyEl){bodyEl.innerHTML=html;scrollToBottom();}
        }

        function handleInputKey(e){
            if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage()}
        }

        function autoResize(el){
            el.style.height='auto';
            el.style.height=Math.min(el.scrollHeight,200)+'px';
        }

        function setStatus(s){document.getElementById('statusBar').textContent=s}

        function copyMsg(idx){
            navigator.clipboard.writeText(state.messages[idx].content||'');
            setStatus('Copied to clipboard');
        }

        async function regenerateMsg(idx){
            if(state.isStreaming)return;
            state.messages=state.messages.slice(0,idx);
            renderMessages();
            await streamResponse();
        }

        function editMsg(idx){
            const m=state.messages[idx];
            const newContent=prompt('Edit message:',m.content);
            if(newContent!==null&&newContent.trim()){
                state.messages=state.messages.slice(0,idx);
                state.messages.push({role:'user',content:newContent.trim(),ts:Date.now()});
                renderMessages();
                streamResponse();
            }
        }

        function showSystemPrompt(){
            document.getElementById('systemPromptText').value=state.systemPrompt;
            document.getElementById('systemPromptModal').style.display='flex';
        }
        function hideSystemPrompt(){document.getElementById('systemPromptModal').style.display='none'}
        function clearSystemPrompt(){state.systemPrompt='';hideSystemPrompt();localStorage.removeItem('novamlx_system_prompt')}
        function saveSystemPrompt(){state.systemPrompt=document.getElementById('systemPromptText').value;localStorage.setItem('novamlx_system_prompt',state.systemPrompt);hideSystemPrompt()}

        function handleFiles(files){
            Array.from(files).forEach(f=>{
                if(f.size>10*1024*1024){setStatus('Image too large (max 10MB)');return}
                const reader=new FileReader();
                reader.onload=e=>{
                    state.images.push(e.target.result);
                    renderImagesPreview();
                };
                reader.readAsDataURL(f);
            });
        }

        function renderImagesPreview(){
            const el=document.getElementById('imagesPreview');
            if(state.images.length===0){el.style.display='none';return}
            el.style.display='flex';
            el.innerHTML=state.images.map((u,i)=>`<div class="img-thumb"><img src="${u}"><button class="remove-img" onclick="removeImage(${i})">&#10005;</button></div>`).join('');
        }

        function removeImage(i){state.images.splice(i,1);renderImagesPreview()}

        function setupDragDrop(){
            const body=document.body;
            body.addEventListener('dragover',e=>{e.preventDefault();state.isDragOver=true});
            body.addEventListener('dragleave',()=>{state.isDragOver=false});
            body.addEventListener('drop',e=>{
                e.preventDefault();state.isDragOver=false;
                const files=e.dataTransfer.files;
                if(files.length)handleFiles(files);
            });
        }

        function setupPaste(){
            document.addEventListener('paste',e=>{
                const items=e.clipboardData?.items;
                if(!items)return;
                for(const item of items){
                    if(item.type.startsWith('image/')){
                        e.preventDefault();
                        handleFiles([item.getAsFile()]);
                        break;
                    }
                }
            });
        }

        function showApiKeyModal(){document.getElementById('apiKeyModal').style.display='flex'}
        function hideApiKeyModal(){document.getElementById('apiKeyModal').style.display='none'}
        function saveApiKey(){apiKey=document.getElementById('apiKeyInput').value.trim();localStorage.setItem('novamlx_api_key',apiKey);hideApiKeyModal();loadModels()}

        init();
        </script>
        </body>
        </html>
        """
    }
}

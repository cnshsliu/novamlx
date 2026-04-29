import Foundation

enum SharedHTML {
    static var css: String {
        """
        *{margin:0;padding:0;box-sizing:border-box}
        :root{--bg:#0a0a0a;--bg2:#141414;--bg3:#1e1e1e;--bg4:#282828;--text:#e5e5e5;--text2:#a3a3a3;--text3:#737373;--border:#262626;--border2:#333;--accent:#8b5cf6;--accent2:#7c3aed;--green:#22c55e;--red:#ef4444;--yellow:#eab308;--blue:#3b82f6;--radius:8px}
        body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);height:100vh;overflow:hidden;display:flex;flex-direction:column}
        .page-container{flex:1;overflow-y:auto;display:none}
        .page-inner{max-width:1100px;margin:0 auto;padding:20px}
        /* Nav */
        .top-nav{height:48px;background:var(--bg2);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 16px;gap:4px;flex-shrink:0}
        .nav-brand{font-weight:700;font-size:15px;margin-right:20px;user-select:none}.nav-brand span{color:var(--accent)}
        .nav-tab{background:none;border:none;color:var(--text2);font-size:13px;padding:8px 14px;cursor:pointer;border-radius:6px;transition:all .15s}
        .nav-tab:hover{color:var(--text);background:var(--bg3)}
        .nav-tab.active{color:var(--accent);background:rgba(139,92,246,.12);font-weight:600}
        .nav-right{margin-left:auto;display:flex;gap:8px;align-items:center}
        .nav-right .btn-sm{font-size:11px;padding:4px 10px}
        /* Buttons */
        .btn-primary{background:var(--accent);color:#fff;border:none;padding:8px 16px;border-radius:var(--radius);cursor:pointer;font-size:13px;font-weight:500;transition:opacity .15s}.btn-primary:hover{opacity:.85}
        .btn-secondary{background:var(--bg3);color:var(--text);border:1px solid var(--border2);padding:8px 16px;border-radius:var(--radius);cursor:pointer;font-size:13px;transition:background .15s}.btn-secondary:hover{background:var(--bg4)}
        .btn-danger{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3);padding:8px 16px;border-radius:var(--radius);cursor:pointer;font-size:13px;transition:background .15s}.btn-danger:hover{background:rgba(239,68,68,.25)}
        .btn-sm{padding:4px 10px;font-size:12px}
        .btn-icon{background:none;border:none;color:var(--text2);cursor:pointer;padding:4px;border-radius:4px;transition:color .15s}.btn-icon:hover{color:var(--text)}
        /* Cards */
        .card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:12px}
        .card-header{font-size:13px;font-weight:600;color:var(--text2);margin-bottom:12px;display:flex;align-items:center;gap:8px}
        .section-card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:16px}
        .section-header{font-size:13px;font-weight:600;color:var(--text2);padding:12px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px}
        .row{padding:10px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px}.row:last-child{border-bottom:none}
        .row:hover{background:var(--bg3)}
        /* Status */
        .status-dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0}
        .status-dot.ok{background:var(--green)}.status-dot.err{background:var(--red)}.status-dot.warn{background:var(--yellow)}
        /* Badge */
        .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:500}
        .badge-green{background:rgba(34,197,94,.15);color:var(--green)}
        .badge-blue{background:rgba(59,130,246,.15);color:var(--blue)}
        .badge-purple{background:rgba(139,92,246,.15);color:var(--accent)}
        .badge-gray{background:var(--bg4);color:var(--text3)}
        /* Inputs */
        .input-field{background:var(--bg3);border:1px solid var(--border2);color:var(--text);padding:8px 12px;border-radius:var(--radius);font-size:13px;outline:none;width:100%;transition:border-color .15s}.input-field:focus{border-color:var(--accent)}
        select.input-field{appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23737373' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center;padding-right:30px}
        /* Modal */
        .modal-overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:1000}
        .modal-card{background:var(--bg2);border:1px solid var(--border2);border-radius:12px;padding:24px;min-width:360px;max-width:90vw;max-height:90vh;overflow-y:auto}
        .modal-card h3{font-size:16px;margin-bottom:16px}
        .modal-actions{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}
        /* Toast */
        #toastContainer{position:fixed;top:56px;right:16px;z-index:2000;display:flex;flex-direction:column;gap:8px}
        .toast{padding:10px 16px;border-radius:var(--radius);font-size:13px;animation:slideIn .2s ease;max-width:320px}
        .toast-info{background:var(--bg3);border:1px solid var(--border2);color:var(--text)}
        .toast-success{background:rgba(34,197,94,.15);border:1px solid rgba(34,197,94,.3);color:var(--green)}
        .toast-error{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.3);color:var(--red)}
        @keyframes slideIn{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:translateX(0)}}
        /* Metric card */
        .metric-card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px}
        .metric-label{font-size:11px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px}
        .metric-value{font-size:22px;font-weight:700;margin-top:4px}
        .metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
        @media(max-width:768px){.metrics-grid{grid-template-columns:repeat(2,1fr)}}
        /* Progress */
        .progress-bar{height:6px;background:var(--bg4);border-radius:3px;overflow:hidden}
        .progress-fill{height:100%;background:var(--accent);border-radius:3px;transition:width .3s}
        /* Placeholder page */
        .placeholder-page{display:flex;flex-direction:column;align-items:center;justify-content:center;height:60vh;opacity:.5}
        .placeholder-icon{font-size:48px;margin-bottom:12px}
        .placeholder-text{font-size:20px;font-weight:600}
        .placeholder-sub{font-size:13px;color:var(--text3);margin-top:4px}
        /* Scrollbar */
        ::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
        /* Mono */
        .mono{font-family:"SF Mono",Monaco,"Cascadia Code",monospace}
        .mono-block{background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);padding:12px;font-family:"SF Mono",Monaco,"Cascadia Code",monospace;font-size:12px;white-space:pre-wrap;overflow-x:auto;line-height:1.5}
        """
    }

    static var navBar: String {
        """
        <nav class="top-nav">
            <div class="nav-brand">Nova<span>MLX</span></div>
            <button class="nav-tab" data-page="status" onclick="navigateTo('status')">Status</button>
            <button class="nav-tab" data-page="models" onclick="navigateTo('models')">Models</button>
            <button class="nav-tab" data-page="chat" onclick="navigateTo('chat')">Chat</button>
            <button class="nav-tab" data-page="agents" onclick="navigateTo('agents')">Agents</button>
            <button class="nav-tab" data-page="settings" onclick="navigateTo('settings')">Settings</button>
            <div class="nav-right">
                <button class="btn-secondary btn-sm" onclick="showApiKeyModal()">API Key</button>
            </div>
        </nav>
        """
    }

    static var js: String {
        """
        const Nova={
            apiKey:localStorage.getItem('novamlx_api_key')||'',
            authHeaders(){return this.apiKey?{'Authorization':'Bearer '+this.apiKey}:{}},
            async api(path,opts={}){
                const headers={'Content-Type':'application/json',...this.authHeaders(),...(opts.headers||{})};
                const resp=await fetch(path,{...opts,headers});
                if(resp.status===401){showApiKeyModal();throw new Error('API key required')}
                if(!resp.ok){const e=await resp.json().catch(()=>({}));throw new Error(e.error?.message||'Server error '+resp.status)}
                return resp;
            },
            formatBytes(b){if(b>=1073741824)return(b/1073741824).toFixed(1)+' GB';if(b>=1048576)return(b/1048576).toFixed(0)+' MB';return(b/1024).toFixed(0)+' KB'},
            formatNumber(n){if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return String(n)},
            formatUptime(s){const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h>0?h+'h '+m+'m':m+'m'},
            escHtml(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
        };

        let currentPage='';
        function navigateTo(page){
            if(!page)page='chat';
            document.querySelectorAll('.page-container').forEach(el=>el.style.display='none');
            const target=document.getElementById('page-'+page);
            if(target)target.style.display='block';
            document.querySelectorAll('.nav-tab').forEach(btn=>btn.classList.toggle('active',btn.dataset.page===page));
            currentPage=page;
            location.hash=page;
            const ns=window[page.charAt(0).toUpperCase()+page.slice(1)+'Page'];
            if(ns&&ns.activate)ns.activate();
            const prev=document.querySelectorAll('.page-container');
            prev.forEach(el=>{if(el.id!=='page-'+page){const n=el.id.replace('page-','');const p=window[n.charAt(0).toUpperCase()+n.slice(1)+'Page'];if(p&&p.deactivate)p.deactivate()}});
        }

        function showApiKeyModal(){document.getElementById('apiKeyModal').style.display='flex';document.getElementById('apiKeyInput').value=Nova.apiKey}
        function hideApiKeyModal(){document.getElementById('apiKeyModal').style.display='none'}
        function saveApiKey(){Nova.apiKey=document.getElementById('apiKeyInput').value.trim();localStorage.setItem('novamlx_api_key',Nova.apiKey);hideApiKeyModal();showToast('API key saved','success')}

        function showToast(msg,type='info'){
            const c=document.getElementById('toastContainer');
            const t=document.createElement('div');t.className='toast toast-'+type;t.textContent=msg;c.appendChild(t);
            setTimeout(()=>t.remove(),3000);
        }

        function copyText(text){navigator.clipboard.writeText(text).then(()=>showToast('Copied','success')).catch(()=>showToast('Copy failed','error'))}
        """
    }
}

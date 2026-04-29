import Foundation

enum AgentsHTML {
    static var content: String {
        """
        <div id="page-agents" class="page-container" style="display:none">
            <div class="page-inner">
                <div class="card">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                        <span class="status-dot ok" id="agentServerDot"></span>
                        <span style="font-weight:600">Server Status</span>
                        <span style="color:var(--text3);font-size:12px" id="agentServerUrl"></span>
                        <button class="btn-secondary btn-sm" style="margin-left:auto" onclick="AgentsPage.refresh()">Refresh</button>
                    </div>
                </div>

                <div class="section-card">
                    <div class="section-header">Available Agents</div>
                    <div id="agentsList"><div class="row" style="color:var(--text3)">Checking...</div></div>
                </div>
            </div>
        </div>
        """
    }

    static var js: String {
        """
        const AgentsPage={
            agents:[],
            activate(){
                document.getElementById('agentServerUrl').textContent=location.host;
                document.getElementById('agentServerDot').className='status-dot ok';
                if(!this._bound){this._bound=true;document.getElementById('page-agents').addEventListener('click',e=>this.handleClick(e))}
                this.refresh();
            },
            deactivate(){},

            handleClick(e){
                const btn=e.target.closest('[data-action]');
                if(!btn)return;
                if(btn.dataset.action==='copy-config'){
                    const idx=parseInt(btn.dataset.idx);
                    const a=this.agents[idx];
                    if(a)copyText(this.generateConfig(a.id));
                }
            },

            async refresh(){
                try{
                    const resp=await Nova.api('/admin/api/agents/check');
                    this.agents=await resp.json();
                    this.render();
                }catch(e){
                    document.getElementById('agentsList').innerHTML='<div class="row" style="color:var(--red)">Failed to check agents: '+Nova.escHtml(e.message)+'</div>';
                }
            },

            render(){
                const el=document.getElementById('agentsList');
                if(!this.agents||!this.agents.length){el.innerHTML='<div class="row" style="color:var(--text3)">No agents configured</div>';return}
                el.innerHTML=this.agents.map((a,i)=>{
                    const installed=a.installed;
                    const cfg=this.generateConfig(a.id);
                    return '<div class="row" style="flex-direction:column;align-items:stretch;gap:8px;padding:14px 16px"><div style="display:flex;align-items:center;gap:10px"><span style="font-size:18px">&#129302;</span><div style="flex:1"><div style="font-weight:600">'+Nova.escHtml(a.name||a.id)+'</div><div style="font-size:12px;color:var(--text3)">'+(installed?'Installed at '+Nova.escHtml(a.path||'unknown'):'Not installed')+'</div></div><span class="badge '+(installed?'badge-green':'badge-gray')+'">'+(installed?'Installed':'Not Found')+'</span>'+(!installed?'<a href="'+Nova.escHtml(a.installUrl||'#')+'" target="_blank" class="btn-primary btn-sm" style="text-decoration:none">Install</a>':'')+'</div><div class="mono-block" style="font-size:11px">'+Nova.escHtml(cfg)+'</div><div><button class="btn-secondary btn-sm" data-action="copy-config" data-idx="'+i+'">Copy Config</button></div></div>';
                }).join('');
            },

            generateConfig(agentId){
                const port=location.port||'6590';
                const key=Nova.apiKey||'(your-api-key)';
                switch(agentId){
                    case 'openclaw':
                        return '// ~/.openclaw/openclaw.json\\n{\\n  "provider": {\\n    "api": "openai-completions",\\n    "baseUrl": "http://127.0.0.1:'+port+'/v1",\\n    "apiKey": "'+key+'"\\n  }\\n}';
                    case 'hermes':
                        return '# ~/.hermes/config.yaml\\nprovider: openai\\nbase_url: http://127.0.0.1:'+port+'/v1\\napi_key: '+key;
                    case 'opencode':
                        return '// Environment variables\\nOPENCODE_PROVIDER=openai\\nOPENCODE_BASE_URL=http://127.0.0.1:'+port+'/v1\\nOPENCODE_API_KEY='+key;
                    default:
                        return 'No config template available';
                }
            }
        };
        """
    }
}

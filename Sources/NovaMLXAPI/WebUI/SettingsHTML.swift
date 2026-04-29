import Foundation

enum SettingsHTML {
    static var content: String {
        """
        <div id="page-settings" class="page-container" style="display:none">
            <div class="page-inner">
                <!-- Server config -->
                <div class="section-card">
                    <div class="section-header">Server Configuration <button class="btn-secondary btn-sm" style="margin-left:auto" onclick="SettingsPage.toggleEdit()" id="configEditBtn">Edit</button></div>
                    <div id="configDisplay" style="padding:12px 16px">
                        <div id="configFields"></div>
                    </div>
                    <div id="configEditor" style="display:none;padding:12px 16px">
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                            <div><label class="metric-label">Host</label><input type="text" class="input-field" id="cfgHost"></div>
                            <div><label class="metric-label">Port</label><input type="number" class="input-field" id="cfgPort" min="1" max="65535"></div>
                            <div><label class="metric-label">Admin Port</label><input type="number" class="input-field" id="cfgAdminPort" min="1" max="65535"></div>
                            <div><label class="metric-label">Max Concurrent</label><input type="number" class="input-field" id="cfgMaxConcurrent" min="1" max="128"></div>
                            <div><label class="metric-label">Timeout (s)</label><input type="number" class="input-field" id="cfgTimeout" min="10" max="3600"></div>
                            <div><label class="metric-label">Max Request Size (MB)</label><input type="number" class="input-field" id="cfgMaxBody" min="1" max="1024"></div>
                            <div style="grid-column:1/-1"><label class="metric-label">API Keys (one per line)</label><textarea class="input-field" id="cfgApiKeys" rows="3" style="resize:vertical"></textarea></div>
                        </div>
                        <div style="display:flex;gap:8px;margin-top:12px;justify-content:flex-end">
                            <button class="btn-secondary" onclick="SettingsPage.toggleEdit()">Cancel</button>
                            <button class="btn-primary" onclick="SettingsPage.saveConfig()">Save</button>
                        </div>
                    </div>
                </div>

                <!-- Sessions -->
                <div class="section-card">
                    <div class="section-header">Sessions <button class="btn-secondary btn-sm" style="margin-left:auto" onclick="SettingsPage.loadSessions()">Refresh</button><button class="btn-danger btn-sm" onclick="SettingsPage.clearSessions()">Clear All</button></div>
                    <div id="sessionsList"><div class="row" style="color:var(--text3)">Click Refresh to load sessions</div></div>
                </div>

                <!-- About -->
                <div class="section-card" id="aboutSection">
                    <div class="section-header">About</div>
                    <div class="row" id="aboutRow"><span style="color:var(--text3)">Loading...</span></div>
                </div>
            </div>
        </div>
        """
    }

    static var js: String {
        """
        const SettingsPage={
            config:null,
            editing:false,

            activate(){this.loadConfig();this.loadAbout()},
            deactivate(){},

            async loadConfig(){
                try{
                    const resp=await Nova.api('/admin/api/config');
                    this.config=await resp.json();
                    this.renderConfig();
                }catch(e){
                    // Fallback: read from health endpoint
                    try{
                        const resp=await Nova.api('/health');
                        const h=await resp.json();
                        this.config={server:{host:'127.0.0.1',port:parseInt(location.port)||6590,adminPort:6591}};
                        this.renderConfig();
                    }catch(e2){}
                }
            },

            renderConfig(){
                if(!this.config)return;
                const s=this.config.server||this.config;
                const fields=[
                    ['API URL','http://127.0.0.1:'+(s.port||6590)],
                    ['Admin URL','http://127.0.0.1:'+(s.adminPort||6591)],
                    ['Web Chat','http://'+location.host+'/#chat'],
                    ['Config','~/.nova/config.json'],
                ];
                document.getElementById('configFields').innerHTML=fields.map(([k,v])=>
                    `<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border);font-size:13px"><span style="color:var(--text3)">${k}</span><span class="mono" style="cursor:pointer" onclick="copyText('${Nova.escHtml(v)}')">${Nova.escHtml(v)}</span></div>`
                ).join('');

                // Fill editor fields
                if(s.host)document.getElementById('cfgHost').value=s.host;
                if(s.port)document.getElementById('cfgPort').value=s.port;
                if(s.adminPort)document.getElementById('cfgAdminPort').value=s.adminPort;
                if(s.maxConcurrentRequests)document.getElementById('cfgMaxConcurrent').value=s.maxConcurrentRequests;
                if(s.requestTimeoutSeconds)document.getElementById('cfgTimeout').value=s.requestTimeoutSeconds;
                if(s.maxRequestSizeMB)document.getElementById('cfgMaxBody').value=s.maxRequestSizeMB;
                if(this.config.apiKeys)document.getElementById('cfgApiKeys').value=this.config.apiKeys.join('\\n');
            },

            toggleEdit(){
                this.editing=!this.editing;
                document.getElementById('configDisplay').style.display=this.editing?'none':'';
                document.getElementById('configEditor').style.display=this.editing?'':'none';
                document.getElementById('configEditBtn').textContent=this.editing?'Cancel':'Edit';
            },

            async saveConfig(){
                const cfg={
                    host:document.getElementById('cfgHost').value,
                    port:parseInt(document.getElementById('cfgPort').value),
                    adminPort:parseInt(document.getElementById('cfgAdminPort').value),
                    maxConcurrentRequests:parseInt(document.getElementById('cfgMaxConcurrent').value),
                    requestTimeoutSeconds:parseInt(document.getElementById('cfgTimeout').value),
                    maxRequestSizeMB:parseFloat(document.getElementById('cfgMaxBody').value),
                };
                const keys=document.getElementById('cfgApiKeys').value.trim().split('\\n').filter(k=>k.trim());
                try{
                    await Nova.api('/admin/api/config',{method:'PUT',body:JSON.stringify({server:cfg,apiKeys:keys})});
                    showToast('Config saved. Restart required.','success');
                    this.editing=false;
                    this.loadConfig();
                }catch(e){showToast('Save failed: '+e.message,'error')}
            },

            async loadSessions(){
                try{
                    const resp=await Nova.api('/admin/sessions');
                    const sessions=await resp.json();
                    const el=document.getElementById('sessionsList');
                    if(!sessions.length){el.innerHTML='<div class="row" style="color:var(--text3)">No active sessions</div>';return}
                    el.innerHTML=sessions.map(s=>`<div class="row">
                        <span style="flex:1;font-size:12px" class="mono">${Nova.escHtml(s.id||s.sessionId||'?')}</span>
                        <span style="color:var(--text3);font-size:12px">${Nova.escHtml(s.modelId||'?')}</span>
                        <button class="btn-icon" style="color:var(--red)" onclick="SettingsPage.deleteSession('${Nova.escHtml(s.id||s.sessionId||'')}')" title="Delete">&times;</button>
                    </div>`).join('');
                }catch(e){showToast('Failed to load sessions: '+e.message,'error')}
            },

            async clearSessions(){
                if(!confirm('Clear all sessions?'))return;
                try{await Nova.api('/admin/sessions',{method:'DELETE'});showToast('Cleared','success');this.loadSessions()}catch(e){showToast(e.message,'error')}
            },

            async deleteSession(id){
                try{await Nova.api('/admin/sessions/'+id,{method:'DELETE'});this.loadSessions()}catch(e){showToast(e.message,'error')}
            },

            async loadAbout(){
                try{
                    const resp=await Nova.api('/admin/api/device-info');
                    const d=await resp.json();
                    document.getElementById('aboutRow').innerHTML=
                        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:13px;width:100%">'+
                        '<div><span style="color:var(--text3)">Version:</span> '+Nova.escHtml(d.novaMLXVersion||d.version||'?')+'</div>'+
                        '<div><span style="color:var(--text3)">Platform:</span> '+Nova.escHtml(d.osVersion||d.platform||'?')+'</div>'+
                        '<div><span style="color:var(--text3)">Chip:</span> '+Nova.escHtml(d.chipName||d.chip||'?')+'</div>'+
                        '<div><span style="color:var(--text3)">Memory:</span> '+(d.memoryGB||d.totalMemoryGB||'?')+' GB</div>'+
                        '<div style="grid-column:1/-1"><a href="https://github.com/cnshsliu/novamlx" target="_blank" style="color:var(--accent)">GitHub</a></div>'+
                        '</div>';
                }catch(e){document.getElementById('aboutRow').innerHTML='<span style="color:var(--text3)">Failed to load</span>'}
            }
        };
        """
    }
}

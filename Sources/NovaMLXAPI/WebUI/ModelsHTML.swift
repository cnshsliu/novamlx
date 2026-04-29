import Foundation

enum ModelsHTML {
    static var content: String {
        """
        <div id="page-models" class="page-container" style="display:none">
            <div class="page-inner">
                <!-- Sub-tabs -->
                <div style="display:flex;gap:4px;margin-bottom:16px">
                    <button class="nav-tab active" data-subtab="my-models" onclick="ModelsPage.switchTab('my-models')">My Models</button>
                    <button class="nav-tab" data-subtab="downloads" onclick="ModelsPage.switchTab('downloads')">Downloads</button>
                </div>

                <!-- My Models tab -->
                <div id="tab-my-models">
                    <div class="section-card" id="loadedModelsSection">
                        <div class="section-header"><span class="status-dot ok"></span> Loaded Models <span class="badge badge-green" id="loadedCount">0</span></div>
                        <div id="loadedModelsList"><div class="row" style="color:var(--text3)">Loading...</div></div>
                    </div>
                    <div class="section-card" id="downloadedModelsSection">
                        <div class="section-header">Downloaded Models <span class="badge badge-gray" id="downloadedCount">0</span></div>
                        <div id="downloadedModelsList"><div class="row" style="color:var(--text3)">Loading...</div></div>
                    </div>
                </div>

                <!-- Downloads tab -->
                <div id="tab-downloads" style="display:none">
                    <!-- Active downloads -->
                    <div class="section-card" id="activeDownloadsSection" style="display:none">
                        <div class="section-header">Active Downloads <span class="badge badge-purple" id="activeDlCount">0</span></div>
                        <div id="activeDownloadsList"></div>
                    </div>

                    <!-- HF Search -->
                    <div class="section-card">
                        <div class="section-header">Search HuggingFace</div>
                        <div style="padding:12px 16px">
                            <div style="display:flex;gap:8px;margin-bottom:8px">
                                <input type="text" class="input-field" id="hfSearchInput" placeholder="Search models..." style="flex:1" onkeydown="if(event.key==='Enter')ModelsPage.search()">
                                <button class="btn-primary" onclick="ModelsPage.search()">Search</button>
                            </div>
                            <div style="display:flex;gap:6px;flex-wrap:wrap" id="suggestedSearches"></div>
                        </div>
                        <div id="searchResults"></div>
                    </div>

                    <!-- Manual download -->
                    <div class="section-card">
                        <div class="section-header">Manual Download</div>
                        <div style="padding:12px 16px;display:flex;gap:8px">
                            <input type="text" class="input-field" id="manualDlInput" placeholder="HuggingFace URL or org/repo" style="flex:1">
                            <button class="btn-primary" onclick="ModelsPage.manualDownload()">Download</button>
                        </div>
                    </div>
                </div>

                <!-- Model card modal -->
                <div id="modelCardModal" class="modal-overlay" style="display:none">
                    <div class="modal-card" style="max-width:600px">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                            <h3 id="modelCardTitle">Model Info</h3>
                            <button class="btn-icon" data-action="close-card" style="font-size:18px">&times;</button>
                        </div>
                        <div id="modelCardBody" style="font-size:13px;line-height:1.6"></div>
                    </div>
                </div>
            </div>
        </div>
        """
    }

    static var js: String {
        """
        const ModelsPage={
            activeTab:'my-models',
            allModels:[],
            searchResults:[],
            pollTimer:null,
            loadedSet:new Set(),

            activate(){
                this.loadModels();
                this.loadSuggested();
                if(!this._bound){this._bound=true;document.getElementById('page-models').addEventListener('click',e=>this.handleClick(e))}
                if(!this.pollTimer)this.pollTimer=setInterval(()=>this.loadModels(),5000);
            },
            deactivate(){
                if(this.pollTimer){clearInterval(this.pollTimer);this.pollTimer=null}
            },

            switchTab(tab){
                this.activeTab=tab;
                document.querySelectorAll('[data-subtab]').forEach(b=>b.classList.toggle('active',b.dataset.subtab===tab));
                document.getElementById('tab-my-models').style.display=tab==='my-models'?'':'none';
                document.getElementById('tab-downloads').style.display=tab==='downloads'?'':'none';
                if(tab==='downloads')this.loadTasks();
            },

            async loadModels(){
                try{
                    const resp=await Nova.api('/admin/models');
                    this.allModels=await resp.json();
                    this.loadedSet=new Set(this.allModels.filter(m=>m.loaded).map(m=>m.id));
                    this.renderMyModels();
                }catch(e){console.warn('loadModels error:',e)}
            },

            renderMyModels(){
                const loaded=this.allModels.filter(m=>m.loaded);
                const downloaded=this.allModels.filter(m=>m.downloaded&&!m.loaded);
                document.getElementById('loadedCount').textContent=loaded.length;
                document.getElementById('downloadedCount').textContent=downloaded.length;

                document.getElementById('loadedModelsList').innerHTML=loaded.length?loaded.map((m,i)=>`
                    <div class="row">
                        <span class="status-dot ok"></span>
                        <span style="flex:1;cursor:pointer" data-action="card" data-idx="${i}">${Nova.escHtml(m.id)}</span>
                        <span class="badge badge-gray">${Nova.escHtml(m.family)}</span>
                        <button class="btn-icon" title="Copy ID" data-action="copy" data-idx="${i}">&#128203;</button>
                        <button class="btn-danger btn-sm" data-action="unload" data-idx="${i}">Unload</button>
                    </div>
                `).join(''):'<div class="row" style="color:var(--text3)">No models loaded</div>';

                document.getElementById('downloadedModelsList').innerHTML=downloaded.length?downloaded.map((m,i)=>`
                    <div class="row">
                        <span style="flex:1;cursor:pointer" data-action="card" data-dlidx="${i}">${Nova.escHtml(m.id)}</span>
                        <span class="badge badge-gray">${Nova.escHtml(m.family)}</span>
                        <span style="color:var(--text3);font-size:12px">${Nova.formatBytes(m.sizeBytes||0)}</span>
                        <button class="btn-primary btn-sm" data-action="load" data-dlidx="${i}">Load</button>
                        <button class="btn-danger btn-sm" data-action="delete" data-dlidx="${i}">Delete</button>
                    </div>
                `).join(''):'<div class="row" style="color:var(--text3)">No downloaded models. Use the Downloads tab to search and download models.</div>';
            },

            handleClick(e){
                const btn=e.target.closest('[data-action]');
                if(!btn)return;
                const action=btn.dataset.action;
                if(action==='card'){
                    const m=btn.dataset.idx!=null?this.allModels.filter(m=>m.loaded)[btn.dataset.idx]:this.allModels.filter(m=>m.downloaded&&!m.loaded)[btn.dataset.dlidx];
                    if(m)this.showModelCard(m.id);
                }else if(action==='search-card'){
                    const m=this.searchResults[btn.dataset.sridx];
                    if(m)this.showModelCard(m.id);
                }else if(action==='close-card'){
                    document.getElementById('modelCardModal').style.display='none';
                }else if(action==='copy'){
                    const m=btn.dataset.idx!=null?this.allModels.filter(m=>m.loaded)[btn.dataset.idx]:null;
                    if(m)copyText(m.id);
                }else if(action==='unload'){
                    const m=this.allModels.filter(m=>m.loaded)[btn.dataset.idx];
                    if(m)this.unload(m.id);
                }else if(action==='load'){
                    const m=this.allModels.filter(m=>m.downloaded&&!m.loaded)[btn.dataset.dlidx];
                    if(m)this.load(m.id);
                }else if(action==='delete'){
                    const m=this.allModels.filter(m=>m.downloaded&&!m.loaded)[btn.dataset.dlidx];
                    if(m)this.deleteModel(m.id);
                }else if(action==='download'){
                    const m=this.searchResults[btn.dataset.sridx];
                    if(m)this.startDownload(m.id);
                }else if(action==='cancel'){
                    this.cancelDownload(btn.dataset.taskid);
                }else if(action==='suggest'){
                    document.getElementById('hfSearchInput').value=btn.dataset.query;
                    this.search();
                }
            },

            async load(id){
                try{
                    await Nova.api('/admin/models/load',{method:'POST',body:JSON.stringify({modelId:id})});
                    showToast('Loading '+id.split('/').pop(),'success');
                    setTimeout(()=>this.loadModels(),2000);
                }catch(e){showToast('Load failed: '+e.message,'error')}
            },
            async unload(id){
                try{
                    await Nova.api('/admin/models/unload',{method:'POST',body:JSON.stringify({modelId:id})});
                    showToast('Unloaded','success');
                    this.loadModels();
                }catch(e){showToast('Unload failed: '+e.message,'error')}
            },
            async deleteModel(id){
                if(!confirm('Delete '+id+'? This removes all downloaded files.'))return;
                try{
                    await Nova.api('/admin/models/'+encodeURIComponent(id),{method:'DELETE'});
                    showToast('Deleted','success');
                    this.loadModels();
                }catch(e){showToast('Delete failed: '+e.message,'error')}
            },

            async search(){
                const q=document.getElementById('hfSearchInput').value.trim();
                if(!q)return;
                document.getElementById('searchResults').innerHTML='<div class="row" style="color:var(--text3)">Searching...</div>';
                try{
                    const resp=await Nova.api('/admin/api/hf/search?q='+encodeURIComponent(q)+'&limit=30');
                    const data=await resp.json();
                    this.searchResults=data.models||[];
                    this.renderSearchResults();
                }catch(e){document.getElementById('searchResults').innerHTML='<div class="row" style="color:var(--red)">Search failed: '+Nova.escHtml(e.message)+'</div>'}
            },

            renderSearchResults(){
                const el=document.getElementById('searchResults');
                if(!this.searchResults.length){el.innerHTML='<div class="row" style="color:var(--text3)">No results</div>';return}
                el.innerHTML=this.searchResults.map((m,i)=>{
                    const dl=this.allModels.find(a=>a.id===m.id);
                    const isDownloaded=dl&&dl.downloaded;
                    const isLoaded=dl&&dl.loaded;
                    let action='';
                    if(isLoaded)action='<span class="badge badge-green">Loaded</span>';
                    else if(isDownloaded)action='<span class="badge badge-blue">Downloaded</span>';
                    else action='<button class="btn-primary btn-sm" data-action="download" data-sridx="'+i+'">Download</button>';
                    return '<div class="row"><span style="flex:1;cursor:pointer" data-action="search-card" data-sridx="'+i+'">'+Nova.escHtml(m.id)+'</span><span style="color:var(--text3);font-size:12px">'+Nova.formatNumber(m.downloads||0)+' downloads</span>'+action+'</div>';
                }).join('');
            },

            async startDownload(repoId){
                try{
                    await Nova.api('/admin/api/hf/download',{method:'POST',body:JSON.stringify({repo_id:repoId})});
                    showToast('Download started: '+repoId.split('/').pop(),'success');
                    this.switchTab('downloads');
                }catch(e){showToast('Download failed: '+e.message,'error')}
            },

            async loadTasks(){
                try{
                    const resp=await Nova.api('/admin/api/hf/tasks');
                    const data=await resp.json();
                    const tasks=data.tasks||[];
                    const active=tasks.filter(t=>t.status==='downloading');
                    document.getElementById('activeDlCount').textContent=active.length;
                    const section=document.getElementById('activeDownloadsSection');
                    if(!active.length){section.style.display='none';return}
                    section.style.display='';
                    document.getElementById('activeDownloadsList').innerHTML=active.map(t=>{
                        const pct=(t.progress||0).toFixed(1);
                        return '<div class="row" style="flex-direction:column;align-items:stretch;gap:8px"><div style="display:flex;align-items:center;gap:8px"><span style="flex:1">'+Nova.escHtml(t.repoId||'unknown')+'</span><span style="color:var(--text3);font-size:12px">'+pct+'%</span><span style="color:var(--text3);font-size:12px">'+Nova.formatBytes(t.downloadedBytes||0)+' / '+Nova.formatBytes(t.totalBytes||0)+'</span><button class="btn-danger btn-sm" data-action="cancel" data-taskid="'+Nova.escHtml(t.taskId||'')+'">Cancel</button></div><div class="progress-bar"><div class="progress-fill" style="width:'+pct+'%"></div></div></div>';
                    }).join('');
                }catch(e){}
            },

            async cancelDownload(taskId){
                try{
                    await Nova.api('/admin/api/hf/cancel',{method:'POST',body:JSON.stringify({task_id:taskId})});
                    showToast('Cancelled','success');
                    this.loadTasks();
                }catch(e){showToast('Cancel failed: '+e.message,'error')}
            },

            async manualDownload(){
                let input=document.getElementById('manualDlInput').value.trim();
                if(!input)return;
                // Parse huggingface URL
                const urlMatch=input.match(/huggingface\\.co\\/([^/]+\\/[^/]+)/);
                if(urlMatch)input=urlMatch[1];
                input=input.replace(/\\.git$/,'');
                if(!input.includes('/')){showToast('Invalid format. Use org/repo or HuggingFace URL','error');return}
                await this.startDownload(input);
                document.getElementById('manualDlInput').value='';
            },

            loadSuggested(){
                const el=document.getElementById('suggestedSearches');
                const suggestions=['mlx-community/Qwen3.6','mlx-community/gemma-4','unsloth/Qwen3','mlx-community/Llama-4'];
                el.innerHTML=suggestions.map(s=>'<button class="btn-secondary btn-sm" data-action="suggest" data-query="'+s+'">'+s+'</button>').join('');
            },

            async showModelCard(repoId){
                document.getElementById('modelCardModal').style.display='flex';
                document.getElementById('modelCardTitle').textContent=repoId;
                document.getElementById('modelCardBody').innerHTML='Loading...';
                try{
                    const resp=await Nova.api('/admin/api/hf/model-info?repo_id='+encodeURIComponent(repoId));
                    const info=await resp.json();
                    let html='<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px">';
                    if(info.downloads!=null)html+=`<div class="metric-card"><div class="metric-label">Downloads</div><div class="metric-value" style="font-size:16px">${Nova.formatNumber(info.downloads)}</div></div>`;
                    if(info.likes!=null)html+=`<div class="metric-card"><div class="metric-label">Likes</div><div class="metric-value" style="font-size:16px">${Nova.formatNumber(info.likes)}</div></div>`;
                    if(info.tags&&info.tags.length)html+=`<div style="grid-column:1/-1;display:flex;gap:4px;flex-wrap:wrap">${info.tags.slice(0,10).map(t=>'<span class="badge badge-gray">'+Nova.escHtml(t)+'</span>').join('')}</div>`;
                    html+='</div>';
                    if(info.library_name)html+='<div style="margin-bottom:6px"><strong>Library:</strong> '+Nova.escHtml(info.library_name)+'</div>';
                    if(info.license)html+='<div style="margin-bottom:6px"><strong>License:</strong> '+Nova.escHtml(info.license)+'</div>';
                    if(info.siblings&&info.siblings.length){
                        html+='<div style="margin-top:8px"><strong>Files:</strong></div><div class="mono-block" style="max-height:200px;overflow-y:auto;font-size:11px">';
                        html+=info.siblings.map(f=>Nova.escHtml(f.rfilename)+' <span style="color:var(--text3)">('+Nova.formatBytes(f.size||0)+')</span>').join('\\n');
                        html+='</div>';
                    }
                    document.getElementById('modelCardBody').innerHTML=html;
                }catch(e){document.getElementById('modelCardBody').innerHTML='<span style="color:var(--red)">Failed: '+Nova.escHtml(e.message)+'</span>'}
            }
        };
        """
    }
}

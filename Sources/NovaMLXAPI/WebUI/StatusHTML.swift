import Foundation

enum StatusHTML {
    static var content: String {
        """
        <div id="page-status" class="page-container" style="display:none">
            <div class="page-inner">
                <!-- Server status hero -->
                <div class="card" id="statusHero">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
                        <span class="status-dot ok" id="statusDot"></span>
                        <span style="font-size:16px;font-weight:600" id="statusText">Online</span>
                        <span style="color:var(--text3);font-size:12px;margin-left:auto" id="statusUrl"></span>
                    </div>
                    <div style="display:flex;gap:16px;font-size:12px;color:var(--text2)">
                        <span>Peak TPS: <strong id="peakTps">--</strong></span>
                    </div>
                </div>

                <!-- TPS Chart -->
                <div class="section-card" style="margin-bottom:16px">
                    <div class="section-header">Real-time Inference Speed <span style="margin-left:auto;font-weight:400;color:var(--text3)" id="currentTps">-- tok/s</span></div>
                    <div style="padding:12px 16px;height:140px">
                        <canvas id="tpsChart"></canvas>
                    </div>
                </div>

                <!-- Metrics grid -->
                <div class="metrics-grid" id="metricsGrid">
                    <div class="metric-card"><div class="metric-label">CPU</div><div class="metric-value" id="mCpu">--</div></div>
                    <div class="metric-card"><div class="metric-label">Memory</div><div class="metric-value" id="mMem">--</div></div>
                    <div class="metric-card"><div class="metric-label">GPU Memory</div><div class="metric-value" id="mGpu">--</div></div>
                    <div class="metric-card"><div class="metric-label">Uptime</div><div class="metric-value" id="mUptime">--</div></div>
                    <div class="metric-card"><div class="metric-label">Active Requests</div><div class="metric-value" id="mRequests">--</div></div>
                    <div class="metric-card"><div class="metric-label">Loaded Models</div><div class="metric-value" id="mModels">--</div></div>
                    <div class="metric-card"><div class="metric-label">Total Tokens</div><div class="metric-value" id="mTokens">--</div></div>
                    <div class="metric-card"><div class="metric-label">Disk Usage</div><div class="metric-value" id="mDisk">--</div></div>
                </div>

                <!-- Device info -->
                <div class="section-card" id="deviceSection">
                    <div class="section-header">Device</div>
                    <div class="row" id="deviceRow">
                        <span style="color:var(--text3)">Loading...</span>
                    </div>
                </div>
            </div>
        </div>
        """
    }

    static var js: String {
        """
        const StatusPage={
            chart:null,
            tpsData:[],
            maxPoints:90,
            peakTps:0,
            pollTimer:null,
            deviceInfo:null,

            activate(){
                this.loadAll();
                this.loadDevice();
                if(!this.pollTimer)this.pollTimer=setInterval(()=>this.loadAll(),2000);
                if(!this.chart)this.initChart();
            },
            deactivate(){
                if(this.pollTimer){clearInterval(this.pollTimer);this.pollTimer=null}
            },

            initChart(){
                const ctx=document.getElementById('tpsChart');
                if(!ctx)return;
                this.chart=new Chart(ctx.getContext('2d'),{
                    type:'line',
                    data:{labels:[],datasets:[{
                        data:[],
                        borderColor:'#8b5cf6',
                        backgroundColor:'rgba(139,92,246,0.08)',
                        fill:true,
                        tension:0.4,
                        pointRadius:0,
                        borderWidth:2,
                    }]},
                    options:{
                        responsive:true,maintainAspectRatio:false,
                        animation:false,
                        plugins:{legend:{display:false},tooltip:{
                            backgroundColor:'#1e1e1e',borderColor:'#333',borderWidth:1,
                            titleFont:{size:11},bodyFont:{size:12},
                            callbacks:{label:ctx=>ctx.parsed.y.toFixed(1)+' tok/s'}
                        }},
                        scales:{
                            x:{display:false},
                            y:{position:'left',ticks:{font:{size:10},color:'#737373'},grid:{color:'rgba(255,255,255,0.04)'},beginAtZero:true}
                        },
                        interaction:{mode:'index',intersect:false}
                    }
                });
            },

            async loadDevice(){
                if(this.deviceInfo)return;
                try{
                    const resp=await Nova.api('/admin/api/device-info');
                    this.deviceInfo=await resp.json();
                    const d=this.deviceInfo;
                    document.getElementById('deviceRow').innerHTML=
                        '<span><strong>'+Nova.escHtml(d.chipName||d.chip||'Unknown')+'</strong>'+(d.chipVariant?' '+Nova.escHtml(d.chipVariant):'')+'</span>'+
                        '<span style="color:var(--text2)">'+(d.memoryGB||d.totalMemoryGB||'?')+' GB</span>'+
                        '<span style="color:var(--text2)">'+(d.gpuCores||d.gpuCoreCount||'?')+' GPU cores</span>'+
                        '<span style="color:var(--text2)">'+(d.cpuCores||d.cpuCoreCount||'?')+' CPU cores</span>';
                }catch(e){}
            },

            async loadAll(){
                try{
                    const [statsR,healthR]=await Promise.all([
                        Nova.api('/v1/stats'),
                        Nova.api('/health')
                    ]);
                    const stats=await statsR.json();
                    const health=await healthR.json();

                    // Status hero
                    const ok=health.status==='ok';
                    document.getElementById('statusDot').className='status-dot '+(ok?'ok':'err');
                    document.getElementById('statusText').textContent=ok?'Online':'Offline';
                    document.getElementById('statusUrl').textContent=location.host;

                    // Metrics
                    const s=stats.session||{};
                    document.getElementById('mCpu').textContent=s.cpuUsage!=null?s.cpuUsage.toFixed(0)+'%':'—';
                    document.getElementById('mMem').textContent=s.memoryUsed!=null?Nova.formatBytes(s.memoryUsed)+' / '+Nova.formatBytes(s.memoryTotal):'—';
                    document.getElementById('mGpu').textContent=Nova.formatBytes(s.gpuMemoryUsed||health.gpuMemoryUsed||0);
                    document.getElementById('mUptime').textContent=health.uptime!=null?Nova.formatUptime(health.uptime):'—';
                    document.getElementById('mRequests').textContent=String(s.activeRequests!=null?s.activeRequests:0);
                    document.getElementById('mModels').textContent=String(health.loadedModels||s.loadedModels||0);
                    document.getElementById('mTokens').textContent=Nova.formatNumber((stats.allTime&&stats.allTime.totalTokens)||0);
                    document.getElementById('mDisk').textContent=health.diskUsage!=null?Nova.formatBytes(health.diskUsage):'—';

                    // TPS — use recentTokensPerSecond (real-time, 5s window) for chart
                    const tps=s.recentTokensPerSecond||0;
                    if(tps>0||this.tpsData.length===0){
                        this.tpsData.push(tps);
                    }else{
                        const zeros=this.tpsData.slice().reverse().findIndex(v=>v>0);
                        if(zeros!==0)this.tpsData.push(tps);
                    }
                    if(this.tpsData.length>this.maxPoints)this.tpsData.shift();
                    if(tps>this.peakTps&&tps<=500)this.peakTps=tps;
                    document.getElementById('peakTps').textContent=this.peakTps.toFixed(1);
                    document.getElementById('currentTps').textContent=tps.toFixed(1)+' tok/s';

                    // Update chart
                    if(this.chart){
                        this.chart.data.labels=this.tpsData.map((_,i)=>i);
                        this.chart.data.datasets[0].data=[...this.tpsData];
                        this.chart.update('none');
                    }
                }catch(e){
                    document.getElementById('statusDot').className='status-dot err';
                    document.getElementById('statusText').textContent='Error';
                }
            }
        };
        """
    }
}

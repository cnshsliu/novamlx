import Foundation
import HTTPTypes
import Hummingbird
import ImageIO
import NovaMLXCore

// MARK: - Admin Proxy & Dashboard
// Extracted from APIServer.swift for modularity.

extension NovaMLXAPIServer {

    static let sessionIDHeader = HTTPField.Name("x-session-id")!

    static func parseQuery(_ query: String) -> [String: String] {
        var result: [String: String] = [:]
        for pair in query.split(separator: "&") {
            let parts = pair.split(separator: "=", maxSplits: 1)
            if parts.count == 2 {
                result[String(parts[0])] = String(parts[1]).removingPercentEncoding ?? String(parts[1])
            } else if parts.count == 1 {
                result[String(parts[0])] = ""
            }
        }
        return result
    }

    static func extractSessionId(request: Request, body: String?) -> String? {
        if let header = request.headers[fields: sessionIDHeader].first?.value, !header.isEmpty {
            return header
        }
        return body
    }

    // MARK: - Admin API Proxy

    static func proxyAdminRequest(path: String, method: String, body: ByteBuffer?, cfg: ServerConfig) async throws -> Response {
        let targetURL = "http://127.0.0.1:\(cfg.adminPort)\(path)"
        guard let url = URL(string: targetURL) else {
            throw NovaMLXError.apiError("Invalid proxy target: \(targetURL)")
        }
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = method
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey = cfg.apiKeys.first {
            urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            urlRequest.httpBody = Data(buffer: body)
        }
        let (data, resp) = try await URLSession.shared.data(for: urlRequest)
        guard let httpResp = resp as? HTTPURLResponse else {
            throw NovaMLXError.apiError("Invalid response from admin server")
        }
        let status = HTTPResponse.Status(code: httpResp.statusCode)
        var headers: HTTPFields = [.contentType: httpResp.value(forHTTPHeaderField: "Content-Type") ?? "application/json"]
        if let cacheControl = httpResp.value(forHTTPHeaderField: "Cache-Control") {
            headers[.cacheControl] = cacheControl
        }
        return Response(status: status, headers: headers, body: .init(byteBuffer: ByteBuffer(data: data)))
    }

    /// Convert raw PNG/JPEG data to a CGImage.
    static func dataToCGImage(_ data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    static func dashboardHTML() -> String {
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NovaMLX Dashboard</title>
        <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0a0a0a;color:#e5e5e5;padding:24px}
        .container{max-width:1200px;margin:0 auto}
        h1{font-size:28px;font-weight:700;margin-bottom:8px;color:#fff}
        h1 span{color:#8b5cf6}
        .subtitle{color:#737373;margin-bottom:32px;font-size:14px}
        .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:24px}
        .card{background:#171717;border:1px solid #262626;border-radius:12px;padding:20px}
        .card h2{font-size:13px;text-transform:uppercase;letter-spacing:0.05em;color:#a3a3a3;margin-bottom:12px}
        .card .value{font-size:32px;font-weight:700;color:#fff}
        .card .sub{font-size:13px;color:#737373;margin-top:4px}
        .status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
        .status-dot.ok{background:#22c55e}
        .status-dot.warn{background:#eab308}
        .status-dot.err{background:#ef4444}
        table{width:100%;border-collapse:collapse;font-size:13px}
        th{text-align:left;color:#737373;padding:8px 12px;border-bottom:1px solid #262626;font-weight:500}
        td{padding:8px 12px;border-bottom:1px solid #1a1a1a}
        .btn{display:inline-block;padding:6px 14px;border-radius:6px;border:1px solid #333;background:#1a1a1a;color:#e5e5e5;font-size:12px;cursor:pointer;margin:2px}
        .btn:hover{background:#262626}
        .btn.danger{border-color:#7f1d1d;color:#fca5a5}
        .btn.danger:hover{background:#7f1d1d}
        .bench-results{margin-top:12px}
        .bench-results table{margin-top:8px}
        #refresh-btn{position:fixed;top:24px;right:24px}
        .nav-links{display:flex;gap:8px;margin-bottom:16px}
        .nav-links a{color:var(--accent);text-decoration:none;font-size:13px;padding:4px 10px;border:1px solid #333;border-radius:6px}
        .nav-links a:hover{background:#1a1a1a}
        @media(max-width:768px){
        body{padding:16px}
        .container{max-width:100%}
        .grid{grid-template-columns:1fr}
        table{font-size:11px}
        th,td{padding:6px 8px}
        #refresh-btn{position:static;margin-bottom:16px}
        }
        </style>
        </head>
        <body>
        <div class="container">
        <h1>Nova<span>MLX</span> Dashboard</h1>
        <p class="subtitle" id="uptime">Loading...</p>
        <div class="nav-links">
        <button class="btn" id="refresh-btn" onclick="loadAll()">Refresh</button>
        <a href="/chat">Chat</a>
        </div>
        <div class="grid" id="cards"></div>
        <div class="card" style="margin-bottom:16px">
        <h2>Device Info</h2>
        <div id="device-info">Loading...</div>
        </div>
        <div class="card" style="margin-bottom:16px">
        <h2>MCP Servers</h2>
        <div id="mcp-info">Loading...</div>
        </div>
        <div class="card" style="margin-bottom:16px">
        <h2>Benchmark</h2>
        <div id="bench-info">
        <button class="btn" onclick="runBench()">Run Benchmark</button>
        <button class="btn" onclick="cancelBench()">Cancel</button>
        <div class="bench-results" id="bench-results"></div>
        </div>
        </div>
        <div class="card" style="margin-bottom:16px">
        <h2>Actions</h2>
        <button class="btn danger" onclick="clearSessionStats()">Clear Session Stats</button>
        <button class="btn danger" onclick="clearAllTimeStats()">Clear All-Time Stats</button>
        </div>
        <div class="card" style="margin-bottom:16px">
        <h2>HuggingFace Model Browser</h2>
        <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">
        <input id="hf-search" type="text" placeholder="Search models (e.g. llama mlx)..." style="flex:1;min-width:200px;padding:8px 12px;background:#1a1a1a;border:1px solid #333;border-radius:6px;color:#e5e5e5;font-size:13px">
        <label style="display:flex;align-items:center;gap:4px;font-size:12px;color:#a3a3a3"><input type="checkbox" id="hf-mlx-only" checked> MLX only</label>
        <button class="btn" onclick="hfSearch()">Search</button>
        </div>
        <div id="hf-results" style="max-height:500px;overflow-y:auto"></div>
        <div id="hf-tasks" style="margin-top:12px"></div>
        </div>
        </div>
        <script>
        const API=location.port==='8081'?'':':8081';
        const BASE='http://'+location.hostname+API;
        const ADMIN_BASE=BASE.replace(/:\\d+/,'');
        let adminToken='';
        function authHeaders(){return adminToken?{'Authorization':'Bearer '+adminToken}:{}}
        async function loadHealth(){
        const r=await fetch(BASE+'/health');
        const d=await r.json();
        document.getElementById('cards').innerHTML=`
        <div class="card"><h2>Status</h2><div class="value"><span class="status-dot ok"></span>${d.status||'ok'}</div><div class="sub">Loaded models: ${d.loadedModels||0}</div></div>
        <div class="card"><h2>GPU Memory</h2><div class="value">${((d.gpuMemoryUsed||0)/1024/1024/1024).toFixed(2)} GB</div><div class="sub">Active GPU allocation</div></div>
        `;
        if(d.mcp){
        const mcpHtml=d.mcp.servers&&d.mcp.servers.length?`<table><tr><th>Server</th><th>State</th><th>Tools</th></tr>${d.mcp.servers.map(s=>`<tr><td>${s.name}</td><td><span class="status-dot ${s.state==='connected'?'ok':'err'}"></span>${s.state}</td><td>${s.toolsCount}</td></tr>`).join('')}</table>`:'<div class="sub">No MCP servers configured</div>';
        document.getElementById('mcp-info').innerHTML=mcpHtml;
        }
        }
        async function loadStats(){
        const r=await fetch(BASE+'/v1/stats');
        const d=await r.json();
        const s=d.session||{};
        const a=d.allTime||{};
        document.getElementById('uptime').textContent=`Session: ${s.totalRequests||0} requests | ${(s.totalTokens||0).toLocaleString()} tokens | ${s.averageTokensPerSecond?.toFixed(1)||0} tok/s | All-time: ${a.totalRequests||0} requests`;
        }
        async function loadDeviceInfo(){
        try{
        const r=await fetch(ADMIN_BASE+':8081/admin/api/device-info',{headers:authHeaders()});
        const d=await r.json();
        document.getElementById('device-info').innerHTML=`<table><tr><th>Chip</th><td>${d.chipName||'N/A'}</td></tr><tr><th>Variant</th><td>${d.chipVariant||'N/A'}</td></tr><tr><th>Memory</th><td>${d.memoryGB||0} GB</td></tr><tr><th>GPU Cores</th><td>${d.gpuCores||0}</td></tr><tr><th>CPU Cores</th><td>${d.cpuCores||0}</td></tr><tr><th>OS</th><td>${d.osVersion||'N/A'}</td></tr><tr><th>NovaMLX</th><td>${d.novaMLXVersion||'N/A'}</td></tr></table>`;
        }catch(e){document.getElementById('device-info').textContent='Admin auth required'}
        }
        async function loadBenchStatus(){
        try{
        const r=await fetch(ADMIN_BASE+':8081/admin/api/bench/status',{headers:authHeaders()});
        const d=await r.json();
        if(d.status==='idle'){document.getElementById('bench-results').innerHTML='<div class="sub">No benchmark running</div>';return}
        let html=`<div class="sub">${d.status} - ${((d.progress||0)*100).toFixed(0)}%</div>`;
        if(d.results&&d.results.length){
        html+='<table><tr><th>Prompt Len</th><th>TTFT (ms)</th><th>Gen tok/s</th><th>Prefill tok/s</th><th>Peak Mem GB</th><th>Latency (s)</th></tr>';
        d.results.forEach(r=>{html+=`<tr><td>${r.prompt_length}</td><td>${r.ttft_ms.toFixed(0)}</td><td>${r.generation_tps.toFixed(1)}</td><td>${r.processing_tps.toFixed(1)}</td><td>${r.peak_memory_gb.toFixed(2)}</td><td>${r.e2e_latency_s.toFixed(2)}</td></tr>`});
        html+='</table>';
        }
        if(d.error)html+=`<div class="sub" style="color:#fca5a5">${d.error}</div>`;
        document.getElementById('bench-results').innerHTML=html;
        }catch(e){document.getElementById('bench-results').innerHTML='<div class="sub">Admin auth required</div>'}
        }
        async function runBench(){
        const model=prompt('Enter model ID to benchmark:');
        if(!model)return;
        try{
        await fetch(ADMIN_BASE+':8081/admin/api/bench/start',{method:'POST',headers:{'Content-Type':'application/json',...authHeaders()},body:JSON.stringify({model_id:model,prompt_lengths:[512,2048,4096],generation_length:128})});
        setTimeout(loadBenchStatus,1000);
        }catch(e){alert('Failed: '+e)}
        }
        async function cancelBench(){
        await fetch(ADMIN_BASE+':8081/admin/api/bench/cancel',{method:'POST',headers:authHeaders()});
        setTimeout(loadBenchStatus,500);
        }
        async function clearSessionStats(){
        await fetch(ADMIN_BASE+':8081/admin/api/stats/clear',{method:'POST',headers:authHeaders()});
        loadStats();
        }
        async function clearAllTimeStats(){
        if(!confirm('Clear all-time stats? This cannot be undone.'))return;
        await fetch(ADMIN_BASE+':8081/admin/api/stats/clear-alltime',{method:'POST',headers:authHeaders()});
        loadStats();
        }
        function loadAll(){loadHealth();loadStats();loadDeviceInfo();loadBenchStatus();loadHFTasks()}
        loadAll();setInterval(function(){loadAll()},5000);
        async function hfSearch(){
        const q=document.getElementById('hf-search').value.trim();
        if(!q)return;
        const mlxOnly=document.getElementById('hf-mlx-only').checked;
        const p=page||1;
        try{
        const url=ADMIN_BASE+':8081/admin/api/hf/search?q='+encodeURIComponent(q)+(mlxOnly?'&mlx_only=true':'')+'&limit=10';
        const r=await fetch(url,{headers:authHeaders()});
        const d=await r.json();
        if(!d.models||!d.models.length){document.getElementById('hf-results').innerHTML='<div class="sub">No models found</div>';return}
        let html='<table><tr><th>Model</th><th>Downloads</th><th>Likes</th><th>Action</th></tr>';
        d.models.forEach(function(m){
        const dl=m.downloads?(m.downloads>1000?(m.downloads/1000).toFixed(1)+'k':m.downloads):'0';
        html+='<tr><td style="max-width:300px;word-break:break-all"><a href="https://huggingface.co/'+m.id+'" target="_blank" style="color:#8b5cf6;text-decoration:none">'+m.id+'</a></td><td>'+dl+'</td><td>'+(m.likes||0)+'</td><td><button class="btn" onclick="hfDownload(\''+m.id+'\')">Download</button></td></tr>';
        });
        html+='</table>';
        document.getElementById('hf-results').innerHTML=html;
        }catch(e){document.getElementById('hf-results').innerHTML='<div class="sub" style="color:#fca5a5">Admin auth required</div>'}
        }
        async function hfDownload(modelId){
        try{
        await fetch(ADMIN_BASE+':8081/admin/api/hf/download',{method:'POST',headers:{'Content-Type':'application/json',...authHeaders()},body:JSON.stringify({repo_id:modelId})});
        loadHFTasks();
        }catch(e){alert('Download failed: '+e)}
        }
        async function loadHFTasks(){
        try{
        const r=await fetch(ADMIN_BASE+':8081/admin/api/hf/tasks',{headers:authHeaders()});
        const d=await r.json();
        if(!d.tasks||!d.tasks.length){document.getElementById('hf-tasks').innerHTML='';return}
        let html='<h3 style="font-size:13px;color:#a3a3a3;margin-bottom:8px">Downloads</h3><table><tr><th>Model</th><th>Progress</th><th>Status</th><th>Action</th></tr>';
        d.tasks.forEach(function(t){
        const pct=t.progress?t.progress.toFixed(0):'0';
        const mb=(t.downloadedBytes/1024/1024).toFixed(0)+'/'+(t.totalBytes/1024/1024).toFixed(0)+'MB';
        html+='<tr><td>'+t.repoId+'</td><td>'+pct+'% ('+mb+')</td><td>'+t.status+'</td><td>'+(t.status==='downloading'||t.status==='pending'?'<button class="btn danger" onclick="hfCancel(\''+t.id+'\')">Cancel</button>':'')+'</td></tr>';
        });
        html+='</table>';
        document.getElementById('hf-tasks').innerHTML=html;
        }catch(e){document.getElementById('hf-tasks').innerHTML=''}
        }
        async function hfCancel(taskId){
        await fetch(ADMIN_BASE+':8081/admin/api/hf/cancel',{method:'POST',headers:{'Content-Type':'application/json',...authHeaders()},body:JSON.stringify({task_id:taskId})});
        loadHFTasks();
        }
        document.getElementById('hf-search').addEventListener('keydown',function(e){if(e.key==='Enter')hfSearch()});
        </script>
        </body>
        </html>
        """
    }

    /// Get MIME type for audio format
    func mimeType(forFormat format: String) -> String {
        switch format.lowercased() {
        case "mp3":
            return "audio/mpeg"
        case "opus":
            return "audio/opus"
        case "aac":
            return "audio/aac"
        case "flac":
            return "audio/flac"
        case "wav":
            return "audio/wav"
        case "aiff":
            return "audio/aiff"
        default:
            return "audio/mpeg"  // Default to MP3
        }
    }
}

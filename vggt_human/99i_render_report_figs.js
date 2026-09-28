#!/usr/bin/env node
/**
 * 99i — 把 report_figs/*.svg 渲染为 3x PNG（3540 宽），供汇报页 / PPT 使用。
 *
 * ⚠️ 坑点：Edge 154 起 `--headless --screenshot=xxx.png` 静默失效 —— 进程 exit 0、
 *   不报错、也不产出文件（连 `--dump-dom` 都无输出）。必须走 CDP：
 *   启动 headless 实例 → WebSocket → Page.captureScreenshot。
 *
 * 用法：
 *   node 99i_render_report_figs.js                  # 渲染全部 4 张
 *   node 99i_render_report_figs.js fig2_solution_architecture 1180 620   # 只渲染一张
 *
 * 输出：report_figs/<name>.png，尺寸 = SVG 的 width/height × 3。
 * 说明：包裹页临时生成在系统 temp 下（含 SVG 内联），渲染完自动清理，不污染仓库。
 */
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const FIGDIR = path.join(__dirname, 'report_figs');
const PORT = 9411;
const SCALE = 3;

const ALL = [
  ['fig1_problem_mapping', 1180, 520],
  ['fig2_solution_architecture', 1180, 620],
  ['fig3_toolchain_dataflow', 1180, 500],
  ['fig4_before_after', 1180, 520],
];
const FIGS = process.argv[2]
  ? [[process.argv[2], +process.argv[3], +process.argv[4]]]
  : ALL;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** 找真实可执行文件：优先版本目录（launcher stub 同样能起，但版本目录更稳） */
function findEdge() {
  const roots = [
    'C:/Program Files (x86)/Microsoft/Edge/Application',
    'C:/Program Files/Microsoft/Edge/Application',
  ];
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    const vers = fs
      .readdirSync(root)
      .filter((d) => /^\d+\.\d+/.test(d) && fs.existsSync(path.join(root, d, 'msedge.exe')))
      .sort((a, b) => b.localeCompare(a, undefined, { numeric: true }));
    if (vers.length) return path.join(root, vers[0], 'msedge.exe');
    const stub = path.join(root, 'msedge.exe');
    if (fs.existsSync(stub)) return stub;
  }
  return null;
}

async function waitReady(timeoutMs) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/version`);
      if (r.ok) return await r.json();
    } catch { /* 还没起来 */ }
    await sleep(300);
  }
  throw new Error('CDP 端口未就绪');
}

/** 单条 CDP 命令（每次连接独立 ws，避免 session 复用问题） */
function rpc(ws, method, params) {
  return new Promise((resolve, reject) => {
    const id = Math.floor(Math.random() * 1e9);
    const onMsg = (ev) => {
      let m;
      try { m = JSON.parse(ev.data); } catch { return; }
      if (m.id !== id) return;
      ws.removeEventListener('message', onMsg);
      m.error ? reject(new Error(`${method}: ${JSON.stringify(m.error)}`)) : resolve(m.result);
    };
    ws.addEventListener('message', onMsg);
    ws.send(JSON.stringify({ id, method, params: params || {} }));
    setTimeout(() => {
      ws.removeEventListener('message', onMsg);
      reject(new Error(`timeout ${method}`));
    }, 30000);
  });
}

async function shot(tmpdir, name, w, h) {
  const svg = fs.readFileSync(path.join(FIGDIR, name + '.svg'), 'utf8');
  // 包裹页：底色由 SVG 自带 style 的 --bg 决定（外层这行会被它覆盖）
  const wrap =
    '<!DOCTYPE html><html><head><meta charset="utf-8">' +
    '<style>html,body{margin:0;padding:0;background:#fff}</style></head><body>\n' +
    svg + '\n</body></html>';
  const page = path.join(tmpdir, name + '.html');
  fs.writeFileSync(page, wrap, 'utf8');

  const url = 'file:///' + page.replace(/\\/g, '/');
  const res = await fetch(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(url)}`, {
    method: 'PUT',
  });
  const target = await res.json();

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve, { once: true });
    ws.addEventListener('error', reject, { once: true });
  });

  await rpc(ws, 'Page.enable');
  await rpc(ws, 'Emulation.setDeviceMetricsOverride', {
    width: w, height: h, deviceScaleFactor: SCALE, mobile: false,
  });
  await rpc(ws, 'Page.navigate', { url });
  await sleep(1200); // 等字体与 SVG 滤镜渲染完
  const { data } = await rpc(ws, 'Page.captureScreenshot', { format: 'png' });

  const out = path.join(FIGDIR, name + '.png');
  fs.writeFileSync(out, Buffer.from(data, 'base64'));
  await rpc(ws, 'Target.closeTarget', { targetId: target.id }).catch(() => {});
  ws.close();
  return (fs.statSync(out).size / 1024).toFixed(0);
}

(async () => {
  const edge = findEdge();
  if (!edge) {
    console.error('❌ 未找到 Edge，请检查安装路径');
    process.exit(1);
  }
  const tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), 'figshot-'));
  const udd = path.join(tmpdir, 'udd');
  console.log(`🚀 [99i] 渲染 ${FIGS.length} 张图 (${SCALE}x)  →  ${FIGDIR}`);
  console.log(`  🌐 ${path.basename(edge)}`);

  const child = spawn(edge, [
    '--headless=new',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${udd}`,
    '--no-first-run', '--no-default-browser-check',
    '--disable-gpu', '--hide-scrollbars',
    'about:blank',
  ], { stdio: 'ignore' });

  let failed = 0;
  try {
    await waitReady(20000);
    for (const [name, w, h] of FIGS) {
      try {
        const kb = await shot(tmpdir, name, w, h);
        console.log(`  ✅ ${name}.png  ${kb} KB  (${w}x${h} @${SCALE}x)`);
      } catch (e) {
        failed++;
        console.error(`  ❌ ${name}: ${e.message}`);
      }
    }
  } catch (e) {
    console.error('❌', e.message);
    failed++;
  } finally {
    child.kill();
    await sleep(500); // 等 Edge 释放 profile 文件句柄，否则 rm 会 EBUSY
    try {
      fs.rmSync(tmpdir, { recursive: true, force: true });
    } catch {
      console.log(`  ⚠️ 临时目录未清理干净（不影响结果）：${tmpdir}`);
    }
  }
  console.log(failed ? '⚠️ 有失败项' : '🎉 Done.');
  process.exit(failed ? 1 : 0);
})();

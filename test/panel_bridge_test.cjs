const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'pages', 'memoir', 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert(script, 'panel script exists');

const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: '', innerHTML: '', hidden: false, textContent: '',
    insertAdjacentHTML(_, value) { this.innerHTML += value; },
  });
  return elements.get(id);
}
const episode = {
  id: 7, title: '真实记忆', content: '今天修好了插件', chat_type: 'private',
  event_start_at: 1, source_raw_ids: '[1]', participants: [], keywords: [],
  raw_evidence: [], raw_missing_ids: [],
};
const calls = [];
const bridge = {
  ready: async () => {},
  async apiGet(route) {
    calls.push(route);
    if (route === 'stats') return {
      episodes_total: 1, episodes_today: 1, raw_total: 1, raw_unprocessed: 0,
      vec_missing: 0, vec_covered: 1, scheduler_running: true,
      embedding_provider_name: 'test', embedding_provider_id: 'test',
      embedding_dim: 8, embedding_status: 'ok', active_sessions: [],
    };
    if (route === 'episodes') return [episode];
    if (route === 'episodes/7') return episode;
    if (route === 'raw') return [{id: 1, created_at: 1, chat_type: 'private',
      role: 'user', speaker_name: '甲', content: '原文'}];
    throw Error(route);
  },
  async apiPost(route) {
    calls.push(route);
    if (route === 'debug/recall') return {query: '测试', scope: {
      resolved_owner_id: '111', current_speaker_id: '111', is_owner: true,
      visibility_scope: 'private_session+all_groups',
    }, results: []};
    if (route === 'vectors/repair') return {repaired: 0, failed: 0};
    throw Error(route);
  },
};
const sandbox = {
  window: {AstrBotPluginPage: bridge},
  document: {getElementById: element, querySelectorAll: () => [], querySelector: () => element('main')},
  Date, Promise, setInterval, clearInterval,
};
vm.runInNewContext(script, sandbox);

(async () => {
  await new Promise(resolve => setImmediate(resolve));
  await sandbox.loadStats();
  assert.match(element('statsGrid').innerHTML, /Episode 总数/);
  await sandbox.loadEpisodes();
  assert.match(element('episodesList').innerHTML, /真实记忆/);
  await sandbox.loadDetail(7);
  assert.match(element('detailContent').innerHTML, /今天修好了插件/);
  await sandbox.loadRaw();
  assert.match(element('rawTableWrap').innerHTML, /原文/);
  element('dbgQuery').value = '测试';
  await sandbox.runDebugRecall();
  assert.match(element('dbgResults').innerHTML, /无命中/);
  assert.match(element('dbgResults').innerHTML, /private_session\+all_groups/);
  assert.match(element('dbgResults').innerHTML, /resolved_owner_id: 111/);
  await sandbox.repairVectors();
  assert.match(element('repairResult').textContent, /补齐 0 条/);
  assert.deepEqual(calls.slice(0, 5), ['stats', 'stats', 'episodes', 'episodes/7', 'raw']);
  console.log('Panel bridge unwrapped data test passed');
})().catch(err => { console.error(err); process.exitCode = 1; });

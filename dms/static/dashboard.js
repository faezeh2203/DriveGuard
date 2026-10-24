// داشبورد درایوگارد — دریافت وضعیت با SSE (در نبود آن، polling)
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const REASON = {
    DROWSY: "خواب‌آلودگی", PHONE: "تلفن همراه", LOOK_DOWN: "نگاه به پایین",
    LOOK_UP: "نگاه به بالا", LATERAL: "نگاه طولانی به اطراف", NO_FACE: "چهره دیده نمی‌شود",
  };
  const KIND = { STARTED: "شروع هشدار", ESCALATED: "هشدار جدی", REASON_CHANGED: "تغییر علت", CLEARED: "رفع شد" };
  const WINDOW_S = 30;
  const LIMITS = { yaw: [-25, 25], pitch: [-15, 20], roll: [-30, 30] };

  // ---------- قالب‌بندی اعداد فارسی ----------
  const nf = {};
  const fa = (v, digits = 0) => {
    const key = String(digits);
    if (!nf[key]) nf[key] = new Intl.NumberFormat("fa-IR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
    return nf[key].format(v);
  };
  const signed = (v, digits = 1) => (v > 0.05 ? "+" : v < -0.05 ? "−" : "") + fa(Math.abs(v), digits) + "°";
  const clock = (s) => {
    s = Math.max(0, Math.floor(s));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
    const p = (n) => fa(n).padStart(2, "۰");
    return (h ? fa(h) + ":" : "") + p(m) + ":" + p(x);
  };

  // ---------- صدای مرورگر ----------
  const alarm = {
    on: false, ctx: null, level: "NONE", timer: null,
    toggle() {
      this.on = !this.on;
      if (this.on && !this.ctx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (AC) this.ctx = new AC();
      }
      if (this.ctx && this.ctx.state === "suspended") this.ctx.resume();
      const b = $("soundBtn");
      b.setAttribute("aria-pressed", String(this.on));
      b.textContent = "صدای مرورگر: " + (this.on ? "روشن" : "خاموش");
      this.set(this.level, true);
    },
    tone(freq, ms) {
      if (!this.ctx) return;
      const o = this.ctx.createOscillator(), g = this.ctx.createGain();
      o.type = "square"; o.frequency.value = freq; g.gain.value = 0.12;
      o.connect(g).connect(this.ctx.destination);
      o.start(); o.stop(this.ctx.currentTime + ms / 1000);
    },
    set(level, force) {
      if (level === this.level && !force) return;
      this.level = level;
      clearInterval(this.timer); this.timer = null;
      if (!this.on) return;
      const spec = level === "CRITICAL" ? [1800, 250, 500] : level === "WARNING" ? [1200, 300, 1000] : null;
      if (!spec) return;
      this.tone(spec[0], spec[1]);
      this.timer = setInterval(() => this.tone(spec[0], spec[1]), spec[2]);
    },
  };
  $("soundBtn").addEventListener("click", () => alarm.toggle());

  // ---------- نوار وضعیت ----------
  function renderStatus(s) {
    const el = $("status");
    const level = s.waiting ? "NONE" : s.alert_level;
    el.dataset.level = level;
    const meter = $("meter"), fill = $("meterFill");
    let label = "وضعیت", title, msg = "", clk = "", ratio = null, recover = false;

    if (s.waiting) {
      title = "در انتظار تصویر";
      msg = "دوربین هنوز فریمی نفرستاده است";
    } else if (s.alert_active) {
      label = level === "CRITICAL" ? "هشدار جدی" : "هشدار";
      title = REASON[s.reason] || "حواس‌پرتی";
      msg = s.message;
      clk = clock(s.alert_seconds);
      if (s.recovery_progress > 0) { ratio = s.recovery_progress; recover = true; msg = "راننده به جاده برگشت؛ در حال تأیید…"; }
    } else if (level === "INFO") {
      title = s.state_fa;
      if (s.threshold_seconds > 0) {
        const left = Math.max(s.threshold_seconds - s.distracted_seconds, 0);
        msg = "هشدار تا " + fa(left, 1) + " ثانیه دیگر";
        ratio = Math.min(s.distracted_seconds / s.threshold_seconds, 1);
      }
      clk = fa(s.distracted_seconds, 1);
    } else {
      title = "عادی";
      msg = "راننده به جاده نگاه می‌کند";
    }
    $("statusLabel").textContent = label;
    $("statusTitle").textContent = title;
    $("statusMsg").textContent = msg;
    $("statusClock").textContent = clk;
    meter.hidden = ratio === null;
    meter.classList.toggle("recover", recover);
    if (ratio !== null) fill.style.width = (ratio * 100).toFixed(1) + "%";
  }

  // ---------- نشانگرها ----------
  function needle(name, v) {
    const pos = (Math.max(-60, Math.min(60, v)) + 60) / 120 * 100;
    const n = $(name + "Needle");
    n.style.left = pos + "%";
    const [lo, hi] = LIMITS[name];
    n.parentElement.classList.toggle("over", v < lo || v > hi);
    $(name).textContent = signed(v);
  }
  function flag(id, text, bad) {
    const el = $(id);
    el.textContent = text;
    el.className = bad ? "bad" : "ok";
  }

  const history = [];
  let lastCount = -1;

  function render(s) {
    renderStatus(s);
    alarm.set(s.waiting ? "NONE" : s.alert_level);
    if (s.waiting) return;

    $("stateName").textContent = s.state_fa;
    needle("yaw", s.yaw); needle("pitch", s.pitch); needle("roll", s.roll);
    $("ear").textContent = s.face_seen ? fa(s.ear, 2) : "—";
    $("perclos").textContent = fa(Math.round((s.perclos || 0) * 100)) + "٪";
    $("blinks").textContent = fa(s.blink_count || 0);
    flag("flagFace", s.face_seen ? "دیده می‌شود" : "دیده نمی‌شود", !s.face_seen);
    flag("flagPhone", s.phone_detected ? "در دست راننده" : "ندارد", s.phone_detected);
    flag("flagEyes", s.drowsy ? "خواب‌آلود" : s.eyes_closed ? "بسته" : "باز", s.drowsy || s.eyes_closed);
    $("fps").textContent = fa(s.fps, 1);
    $("latency").textContent = fa(s.latency_ms, 0) + " میلی‌ثانیه";
    $("alertCount").textContent = fa(s.alert_count || 0);
    $("uptime").textContent = clock(s.uptime || 0);

    history.push({ t: s.ts, yaw: s.yaw, pitch: s.pitch, roll: s.roll, alert: s.alert_active });
    while (history.length && history[0].t < s.ts - WINDOW_S) history.shift();
    if (s.alert_count !== lastCount || s.alert_active !== render.wasActive) {
      lastCount = s.alert_count; render.wasActive = s.alert_active; loadLog();
    }
  }

  // ---------- نمودار ----------
  const cv = $("chart"), g = cv.getContext("2d");
  const COL = { yaw: "#e4e1da", pitch: "#e0a03a", roll: "#6f6c66", line: "#2a2c2f", dim: "#8d8a83" };
  const NAMES = { yaw: "افقی", pitch: "عمودی", roll: "کج" };
  let hatch = null;

  function makeHatch() {
    const c = document.createElement("canvas"); c.width = c.height = 8;
    const x = c.getContext("2d");
    x.strokeStyle = "rgba(210,75,63,.45)"; x.lineWidth = 1;
    x.beginPath(); x.moveTo(0, 8); x.lineTo(8, 0); x.stroke();
    return g.createPattern(c, "repeat");
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); hatch = null; }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    if (!hatch) hatch = makeHatch();
    const L = 30, R = 44, T = 6, B = 6, R0 = 60;
    const y = (v) => T + (h - T - B) * (1 - (Math.max(-R0, Math.min(R0, v)) + R0) / (2 * R0));
    g.font = "11px Vazirmatn, Tahoma, sans-serif"; g.textBaseline = "middle";
    for (const v of [-60, -30, 0, 30, 60]) {
      g.strokeStyle = COL.line; g.lineWidth = 1;
      g.setLineDash(v === 0 ? [] : [2, 4]);
      g.beginPath(); g.moveTo(L, Math.round(y(v)) + .5); g.lineTo(w - R, Math.round(y(v)) + .5); g.stroke();
      g.fillStyle = COL.dim; g.textAlign = "left"; g.fillText(fa(v), 2, y(v));
    }
    g.setLineDash([]);
    if (history.length < 2) return;
    const t1 = history[history.length - 1].t, t0 = t1 - WINDOW_S;
    const x = (t) => L + (w - L - R) * ((t - t0) / WINDOW_S);
    g.fillStyle = hatch;
    for (let i = 1; i < history.length; i++) {
      if (history[i].alert) g.fillRect(x(history[i - 1].t), T, x(history[i].t) - x(history[i - 1].t) + .5, h - T - B);
    }
    for (const k of ["roll", "pitch", "yaw"]) {
      g.strokeStyle = COL[k]; g.lineWidth = k === "roll" ? 1 : 1.5; g.beginPath();
      history.forEach((p, i) => (i ? g.lineTo(x(p.t), y(p[k])) : g.moveTo(x(p.t), y(p[k]))));
      g.stroke();
      const last = history[history.length - 1];
      g.fillStyle = COL[k]; g.textAlign = "left"; g.fillText(NAMES[k], w - R + 6, y(last[k]));
    }
  }
  (function loop() { draw(); requestAnimationFrame(loop); })();

  // ---------- جدول رخدادها ----------
  async function loadLog() {
    let rows;
    try { rows = await (await fetch("/api/alerts")).json(); } catch (e) { return; }
    const body = $("log");
    body.textContent = "";
    if (!rows.length) {
      const tr = document.createElement("tr"); tr.className = "empty";
      const td = document.createElement("td"); td.colSpan = 4; td.textContent = "رخدادی ثبت نشده";
      tr.appendChild(td); body.appendChild(tr); return;
    }
    for (const r of rows) {
      const tr = document.createElement("tr");
      const cells = [
        clock(r.ts),
        KIND[r.kind] || r.kind,
        REASON[r.reason] || r.state,
        r.kind === "CLEARED" ? fa(r.duration, 1) + " ث" : "—",
      ];
      cells.forEach((text, i) => {
        const td = document.createElement("td");
        td.textContent = text;
        if (i === 1) td.className = "k-" + r.kind;
        tr.appendChild(td);
      });
      body.appendChild(tr);
    }
  }

  // ---------- اتصال ----------
  const conn = $("conn");
  const online = (ok) => {
    conn.classList.toggle("on", ok);
    conn.lastElementChild.textContent = ok ? "متصل" : "قطع ارتباط";
  };

  if (window.EventSource) {
    // وضعیت اتصال از روی رسیدن داده تعیین می‌شود، نه خطای لحظه‌ای EventSource
    // (مرورگر خودش دوباره وصل می‌شود)
    let lastMsg = 0;
    const es = new EventSource("/api/stream");
    es.onmessage = (ev) => { lastMsg = Date.now(); online(true); render(JSON.parse(ev.data)); };
    setInterval(() => { if (Date.now() - lastMsg > 3000) online(false); }, 1000);
  } else {
    setInterval(async () => {
      try { render(await (await fetch("/api/status")).json()); online(true); }
      catch (e) { online(false); }
    }, 300);
  }
  loadLog();
})();

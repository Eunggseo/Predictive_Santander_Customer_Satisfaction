import pptxgen from "/Users/wenyuzhong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs/dist/pptxgen.es.js";

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.company = "Predictive Analytics";
pptx.subject = "Santander customer dissatisfaction predictive modeling";
pptx.title = "Identifying dissatisfied customers early";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};

const slide = pptx.addSlide();
slide.background = { color: "F7F5F2" };

const C = {
  red: "C8102E",
  redDark: "8A0E1A",
  ink: "1F2933",
  muted: "5F6875",
  faint: "E8E1DA",
  box: "FFFFFF",
  label: "6B7280",
};

const W = 13.333;
const H = 7.5;

function addText(text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h,
    margin: 0,
    fontFace: opts.fontFace || "Aptos",
    fontSize: opts.fontSize || 16,
    bold: opts.bold || false,
    color: opts.color || C.ink,
    breakLine: false,
    fit: "shrink",
    valign: opts.valign || "top",
    align: opts.align || "left",
    paraSpaceAfterPt: 0,
    paraSpaceBeforePt: 0,
    ...opts,
  });
}

function addLine(x1, y1, x2, y2, opts = {}) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color: opts.color || C.red, width: opts.width || 2, beginArrowType: opts.beginArrowType, endArrowType: opts.endArrowType },
  });
}

function addArrow(x, y, w) {
  addLine(x, y, x + w, y, { color: C.red, width: 3.2, endArrowType: "triangle" });
}

function iconChurn(cx, cy) {
  slide.addShape(pptx.ShapeType.arc, { x: cx - 0.22, y: cy - 0.22, w: 0.26, h: 0.26, adjustPoint: 0.4, line: { color: C.red, width: 1.5 }, fill: { color: "FFFFFF", transparency: 100 } });
  slide.addShape(pptx.ShapeType.ellipse, { x: cx - 0.27, y: cy - 0.28, w: 0.18, h: 0.18, line: { color: C.red, width: 1.5 }, fill: { color: "FFFFFF", transparency: 100 } });
  slide.addShape(pptx.ShapeType.arc, { x: cx - 0.36, y: cy - 0.08, w: 0.36, h: 0.24, adjustPoint: 0.5, line: { color: C.red, width: 1.5 }, fill: { color: "FFFFFF", transparency: 100 } });
  addLine(cx + 0.05, cy - 0.08, cx + 0.40, cy - 0.08, { color: C.red, width: 1.7, endArrowType: "triangle" });
}

function iconData(cx, cy) {
  const bars = [[0.13, 0.33], [0.26, 0.22], [0.39, 0.43]];
  bars.forEach(([dx, bh]) => {
    slide.addShape(pptx.ShapeType.rect, { x: cx - 0.34 + dx, y: cy + 0.16 - bh, w: 0.08, h: bh, line: { color: C.red, transparency: 100 }, fill: { color: C.red } });
  });
  addLine(cx - 0.28, cy + 0.16, cx + 0.28, cy + 0.16, { color: C.redDark, width: 1.2 });
  addLine(cx - 0.28, cy + 0.16, cx - 0.28, cy - 0.28, { color: C.redDark, width: 1.2 });
  slide.addShape(pptx.ShapeType.arc, { x: cx + 0.05, y: cy - 0.30, w: 0.28, h: 0.28, adjustPoint: 0.2, line: { color: C.redDark, width: 1.2 }, fill: { color: "FFFFFF", transparency: 100 } });
}

function iconTarget(cx, cy) {
  [0.58, 0.38, 0.18].forEach(s => {
    slide.addShape(pptx.ShapeType.ellipse, { x: cx - s / 2, y: cy - s / 2, w: s, h: s, line: { color: s === 0.18 ? C.red : C.redDark, width: s === 0.18 ? 1.8 : 1.2 }, fill: { color: "FFFFFF", transparency: 100 } });
  });
  addLine(cx - 0.36, cy + 0.34, cx + 0.22, cy - 0.24, { color: C.red, width: 1.8, endArrowType: "triangle" });
}

function addCard({ x, label, title, metric, metricSub, body, icon }) {
  addText(label, x, 1.78, 3.35, 0.16, {
    fontSize: 8.8,
    bold: true,
    color: C.label,
    charSpace: 1.4,
    align: "center",
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y: 2.02, w: 3.35, h: 3.72,
    rectRadius: 0.06,
    line: { color: C.faint, width: 1.1 },
    fill: { color: C.box },
    shadow: { type: "outer", color: "D6D0CA", opacity: 0.17, blur: 1, angle: 45, distance: 1 },
  });
  icon(x + 1.675, 2.52);
  addText(title, x + 0.28, 2.86, 2.79, 0.33, {
    fontSize: 16.2,
    bold: true,
    color: C.ink,
    align: "center",
    fontFace: "Aptos Display",
  });
  addText(metric, x + 0.28, 3.34, 2.79, 0.45, {
    fontSize: 24.5,
    bold: true,
    color: C.red,
    align: "center",
    fontFace: "Aptos Display",
  });
  addText(metricSub, x + 0.28, 3.77, 2.79, 0.21, {
    fontSize: 8.8,
    bold: true,
    color: C.redDark,
    align: "center",
    charSpace: 0.5,
  });
  addText(body, x + 0.38, 4.28, 2.59, 0.84, {
    fontSize: 12.8,
    color: C.muted,
    align: "center",
    valign: "mid",
    fit: "shrink",
    breakLine: false,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: x + 0.58, y: 5.32, w: 2.19, h: 0,
    line: { color: C.faint, width: 1 },
  });
}

slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: W, h: 0.14, line: { color: C.red, transparency: 100 }, fill: { color: C.red } });
slide.addShape(pptx.ShapeType.rect, { x: 0, y: H - 0.09, w: W, h: 0.09, line: { color: C.redDark, transparency: 100 }, fill: { color: C.redDark } });

addText("Identifying dissatisfied customers early is critical", 0.72, 0.52, 7.45, 0.47, {
  fontFace: "Aptos Display",
  fontSize: 24,
  bold: true,
  color: C.ink,
});
addText("but the warning signals are hidden in complex, high-dimensional data", 0.72, 1.02, 8.6, 0.32, {
  fontSize: 13.8,
  color: C.muted,
});
addText("SANTANDER CUSTOMER DISSATISFACTION", 9.28, 0.60, 3.26, 0.20, {
  fontSize: 8.6,
  bold: true,
  color: C.redDark,
  align: "right",
  charSpace: 1.5,
});

const xs = [0.85, 4.99, 9.13];
addCard({
  x: xs[0],
  label: "THE CONTEXT",
  title: "Retention is strategic",
  metric: "early signals",
  metricSub: "BEFORE CHURN SHOWS UP",
  body: "Dissatisfied customers often leave without complaining first — by the time churn appears, intervention is late.",
  icon: iconChurn,
});
addCard({
  x: xs[1],
  label: "THE CHALLENGE",
  title: "Detection is hard",
  metric: "76K | 306 | 3.96%",
  metricSub: "CUSTOMERS  FEATURES  DISSATISFIED",
  body: "Sparse labels and anonymized features make manual rules unreliable and simple analysis too shallow.",
  icon: iconData,
});
addCard({
  x: xs[2],
  label: "OUR APPROACH",
  title: "Predict risk early",
  metric: "proactive retention",
  metricSub: "MODEL-LED ACTION",
  body: "Identify high-risk customers early so Santander can turn passive churn into targeted retention.",
  icon: iconTarget,
});

addArrow(4.27, 3.86, 0.52);
addArrow(8.41, 3.86, 0.52);

addText("Predictive analytics converts silent dissatisfaction into an actionable retention queue.", 1.42, 6.32, 10.50, 0.30, {
  fontSize: 12.4,
  bold: true,
  color: C.ink,
  align: "center",
});

await pptx.writeFile({ fileName: "/Users/wenyuzhong/Desktop/Predictive Analytics/project/santander_customer_dissatisfaction_slide.pptx" });

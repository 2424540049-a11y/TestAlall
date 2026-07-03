const params = new URLSearchParams(window.location.search);
const rawUrl = params.get("url") || "";
const title = params.get("title") || "新闻详情";
const source = params.get("source") || "";
const time = params.get("time") || "";
const description = params.get("description") || "";
const descriptionZh = params.get("descriptionZh") || "";
const titleZh = params.get("titleZh") || "";

const els = {
  title: document.querySelector("#readerTitle"),
  meta: document.querySelector("#readerMeta"),
  url: document.querySelector("#readerUrl"),
  frame: document.querySelector("#articleFrame"),
  origin: document.querySelector("#originLink"),
  status: document.querySelector("#readerStatus"),
  summary: document.querySelector("#articleSummary"),
  summaryText: document.querySelector("#summaryText"),
  summaryZhText: document.querySelector("#summaryZhText"),
  embedNotice: document.querySelector("#embedNotice"),
  back: document.querySelector("#backButton")
};

function safeUrl(value) {
  try {
    const url = new URL(value);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    return url;
  } catch {
    return null;
  }
}

function openOriginal(event) {
  event.preventDefault();
  const popup = window.open(els.origin.href, "_blank", "noopener,noreferrer");
  if (popup) popup.opener = null;
  else window.location.assign(els.origin.href);
}

const targetUrl = safeUrl(rawUrl);
els.title.textContent = title;
document.title = title;
els.meta.textContent = [source, time].filter(Boolean).join(" · ") || "新闻详情";

const summaryText = description || title;
const summaryZhText = descriptionZh || titleZh;
if (summaryText || summaryZhText) {
  els.summary.hidden = false;
  els.summaryText.textContent = summaryText;
  els.summaryText.hidden = !summaryText;
  els.summaryZhText.textContent = summaryZhText ? `中文翻译：${summaryZhText}` : "";
  els.summaryZhText.hidden = !summaryZhText;
}

els.back.addEventListener("click", () => {
  if (window.history.length > 1) window.history.back();
  else window.location.assign("/");
});

if (targetUrl) {
  els.url.textContent = targetUrl.hostname;
  els.origin.href = targetUrl.toString();
  els.origin.addEventListener("click", openOriginal);
  const canEmbed =
    targetUrl.hostname.endsWith("smm.cn") || targetUrl.hostname.endsWith("shfe.com.cn");
  if (canEmbed) {
    els.frame.src = targetUrl.toString();
  } else {
    els.frame.hidden = true;
    els.summary.hidden = false;
    els.embedNotice.hidden = false;
  }
} else {
  els.url.textContent = "--";
  els.origin.hidden = true;
  els.frame.hidden = true;
  els.status.hidden = false;
  els.status.textContent = "链接不可用。";
}

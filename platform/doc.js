const params = new URLSearchParams(window.location.search);
const source = params.get("src") || "../docs/final_report_full_score.md";
const title = params.get("title") || "平台文档";

const docTitle = document.querySelector("#docTitle");
const rawLink = document.querySelector("#rawLink");
const docContent = document.querySelector("#docContent");

docTitle.textContent = title;
document.title = `${title} - 平台文档查看器`;
rawLink.href = source;
rawLink.download = source.split("/").pop() || "document.md";

const escapeHtml = (value) =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const escapeAttr = (value) => escapeHtml(value).replace(/`/g, "&#96;");

const renderInline = (value) => {
  let html = escapeHtml(value);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => {
    return `<a href="${escapeAttr(href)}" target="_blank" rel="noreferrer">${label}</a>`;
  });
  return html;
};

const isTableSeparator = (line) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);

function parseTable(lines, startIndex) {
  const rows = [];
  let index = startIndex;
  while (index < lines.length && lines[index].includes("|") && lines[index].trim() !== "") {
    if (!isTableSeparator(lines[index])) {
      const cells = lines[index]
        .trim()
        .replace(/^\|/, "")
        .replace(/\|$/, "")
        .split("|")
        .map((cell) => renderInline(cell.trim()));
      rows.push(cells);
    }
    index += 1;
  }

  if (rows.length === 0) return { html: "", nextIndex: index };
  const [header, ...bodyRows] = rows;
  const head = `<thead><tr>${header.map((cell) => `<th>${cell}</th>`).join("")}</tr></thead>`;
  const body = `<tbody>${bodyRows
    .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
    .join("")}</tbody>`;
  return { html: `<div class="doc-table-wrap"><table>${head}${body}</table></div>`, nextIndex: index };
}

function renderList(items, ordered = false) {
  const tag = ordered ? "ol" : "ul";
  return `<${tag}>${items.map((item) => `<li>${renderInline(item)}</li>`).join("")}</${tag}>`;
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let paragraph = [];
  let index = 0;

  const flushParagraph = () => {
    if (paragraph.length > 0) {
      blocks.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (trimmed === "") {
      flushParagraph();
      index += 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      flushParagraph();
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      index += 1;
      continue;
    }

    if (/^#{1,4}\s+/.test(trimmed)) {
      flushParagraph();
      const level = Math.min(trimmed.match(/^#+/)[0].length, 4);
      const text = trimmed.replace(/^#{1,4}\s+/, "");
      blocks.push(`<h${level}>${renderInline(text)}</h${level}>`);
      index += 1;
      continue;
    }

    if (/^---+$/.test(trimmed)) {
      flushParagraph();
      blocks.push("<hr />");
      index += 1;
      continue;
    }

    if (trimmed.includes("|") && index + 1 < lines.length && isTableSeparator(lines[index + 1])) {
      flushParagraph();
      const table = parseTable(lines, index);
      blocks.push(table.html);
      index = table.nextIndex;
      continue;
    }

    if (/^>\s+/.test(trimmed)) {
      flushParagraph();
      const quoteLines = [];
      while (index < lines.length && /^>\s+/.test(lines[index].trim())) {
        quoteLines.push(lines[index].trim().replace(/^>\s+/, ""));
        index += 1;
      }
      blocks.push(`<blockquote>${quoteLines.map((item) => `<p>${renderInline(item)}</p>`).join("")}</blockquote>`);
      continue;
    }

    if (/^[-*]\s+/.test(trimmed)) {
      flushParagraph();
      const items = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^[-*]\s+/, ""));
        index += 1;
      }
      blocks.push(renderList(items));
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      flushParagraph();
      const items = [];
      while (index < lines.length && /^\d+\.\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^\d+\.\s+/, ""));
        index += 1;
      }
      blocks.push(renderList(items, true));
      continue;
    }

    paragraph.push(trimmed);
    index += 1;
  }

  flushParagraph();
  return blocks.join("\n");
}

async function loadMarkdown() {
  try {
    const response = await fetch(source, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    const markdown = new TextDecoder("utf-8").decode(buffer);
    docContent.innerHTML = renderMarkdown(markdown);
  } catch (error) {
    docContent.innerHTML = `
      <div class="doc-error">
        <h2>文档读取失败</h2>
        <p>无法打开 <code>${escapeHtml(source)}</code>，请确认本地预览服务器仍在运行。</p>
        <small>${escapeHtml(error.message)}</small>
      </div>
    `;
  }
}

loadMarkdown();

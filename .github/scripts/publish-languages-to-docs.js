const fs = require("fs");

const scripts = {};
for (const entry of fs.readdirSync(".", { withFileTypes: true })) {
  if (!entry.isDirectory() || !entry.name.startsWith("jg-")) continue;
  const locales = fs
    .readdirSync(entry.name)
    .filter((f) => f.endsWith(".lua"))
    .map((f) => f.replace(/\.lua$/, ""))
    .sort();
  if (locales.length) scripts[entry.name] = locales;
}

const payload = {
  event_type: "languages",
  client_payload: { scripts },
};

fetch("https://api.github.com/repos/jgscripts/jg-docs/dispatches", {
  method: "POST",
  headers: {
    authorization: `Bearer ${process.env.GH_API_TOKEN}`,
    accept: "application/vnd.github+json",
    "user-agent": "jg-translations-workflow",
  },
  body: JSON.stringify(payload),
})
  .then(async (res) => {
    if (!res.ok) {
      console.error("Languages dispatch failed:", res.status, await res.text());
      process.exit(1);
    }
    console.log(
      "Languages payload dispatched to jgscripts/jg-docs:",
      Object.entries(scripts)
        .map(([k, v]) => `${k} (${v.length})`)
        .join(", ")
    );
  })
  .catch((err) => {
    console.error("Languages dispatch failed:", err);
    process.exit(1);
  });

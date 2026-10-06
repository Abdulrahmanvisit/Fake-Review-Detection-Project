const { spawn } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const chromePath = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const outputDirectory = __dirname;
const profileDirectory = path.join(
  os.tmpdir(),
  `fake-review-chapter4-profile-${Date.now()}`,
);
const debuggingPort = 9227;
const applicationUrl = "http://127.0.0.1:3015/";

const chrome = spawn(
  chromePath,
  [
    "--headless=new",
    "--disable-gpu",
    "--no-first-run",
    "--no-default-browser-check",
    "--hide-scrollbars",
    `--user-data-dir=${profileDirectory}`,
    `--remote-debugging-port=${debuggingPort}`,
    "--window-size=1440,900",
    "--force-device-scale-factor=1",
    applicationUrl,
  ],
  { stdio: "ignore" },
);

let socket;
let nextMessageId = 0;
let pendingApiResponse;
const pendingMessages = new Map();

function send(method, params = {}) {
  const id = ++nextMessageId;
  return new Promise((resolve, reject) => {
    pendingMessages.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function waitForDebugger() {
  const endpoint = `http://127.0.0.1:${debuggingPort}/json/list`;
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      const response = await fetch(endpoint);
      if (response.ok) {
        const targets = await response.json();
        const target = targets.find(
          (item) => item.type === "page" && item.url.startsWith(applicationUrl),
        );
        if (target) return target;
      }
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("Chrome DevTools did not become available.");
}

async function evaluate(expression) {
  const response = await send("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (response.exceptionDetails) {
    throw new Error(response.exceptionDetails.text);
  }
  return response.result?.value;
}

async function waitForResults() {
  await send("Network.enable");
  const responsePromise = new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pendingApiResponse = undefined;
      reject(new Error("Timed out waiting for the /api/predict response."));
    }, 45000);
    pendingApiResponse = (status) => {
      clearTimeout(timeout);
      pendingApiResponse = undefined;
      resolve(status);
    };
  });

  await evaluate("document.querySelector('.analyze-button').click()");
  const status = await responsePromise;
  if (status !== 200) {
    throw new Error(`Prediction request returned HTTP ${status}.`);
  }

  await evaluate(`new Promise((resolve, reject) => {
    const started = Date.now();
    const check = () => {
      const results = document.querySelector('[aria-labelledby="results-heading"]');
      if (results?.innerText.toLowerCase().includes('prediction confidence')) {
        resolve(true);
      } else if (Date.now() - started > 10000) {
        reject(new Error('The results UI did not render the submitted review.'));
      } else {
        requestAnimationFrame(check);
      }
    };
    check();
  })`);
}

async function enterReview(review, rating, verifiedPurchase) {
  await evaluate(`(() => {
    const field = document.querySelector('#review');
    field.focus();
    field.select();
  })()`);
  await send("Input.insertText", { text: review });
  await evaluate(`(() => {
    const rating = document.querySelector('#rating');
    rating.value = ${JSON.stringify(String(rating))};
    rating.dispatchEvent(new Event('change', { bubbles: true }));
    const verified = document.querySelector('#verified');
    verified.value = ${JSON.stringify(verifiedPurchase)};
    verified.dispatchEvent(new Event('change', { bubbles: true }));
  })()`);
}

async function submitReview() {
  await waitForResults();
}

async function saveViewportScreenshot(fileName) {
  await evaluate("window.scrollTo(0, 0)");
  const capture = await send("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    captureBeyondViewport: false,
  });
  fs.writeFileSync(
    path.join(outputDirectory, fileName),
    Buffer.from(capture.data, "base64"),
  );
}

async function savePipelineScreenshot(fileName) {
  const bounds = await evaluate(`(() => {
    const section = document.querySelector('[aria-labelledby="pipeline-heading"]');
    const rect = section.getBoundingClientRect();
    return { x: rect.x, y: rect.y + window.scrollY, width: rect.width, height: rect.height };
  })()`);
  const capture = await send("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    captureBeyondViewport: true,
    clip: { ...bounds, scale: 1 },
  });
  fs.writeFileSync(
    path.join(outputDirectory, fileName),
    Buffer.from(capture.data, "base64"),
  );
}

async function main() {
  const target = await waitForDebugger();
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (
      message.method === "Network.responseReceived" &&
      message.params?.response?.url?.endsWith("/api/predict")
    ) {
      pendingApiResponse?.(message.params.response.status);
    }
    if (message.id === undefined) return;
    const pending = pendingMessages.get(message.id);
    if (!pending) return;
    pendingMessages.delete(message.id);
    if (message.error) pending.reject(new Error(message.error.message));
    else pending.resolve(message.result);
  });

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await send("Emulation.setPageScaleFactor", { pageScaleFactor: 1 });
  await evaluate(`new Promise(resolve => {
    if (document.readyState === 'complete') resolve(true);
    else window.addEventListener('load', () => resolve(true), { once: true });
  })`);

  await saveViewportScreenshot("Figure 4.1 - Home Interface.png");

  await evaluate(`[...document.querySelectorAll('button')]
    .find(button => button.textContent.trim() === 'Use example')?.click()`);
  await saveViewportScreenshot("Figure 4.2 - Review Input.png");

  await enterReview(
    "The product quality was poor, the delivery was late, the packaging was damaged, and the price was too expensive.",
    4,
    "Y",
  );
  await submitReview();
  const negativeResult = await evaluate(`(() => {
    const region = document.querySelector('[aria-labelledby="results-heading"]');
    return { text: region.innerText, aspects: document.querySelectorAll('.aspect-row').length };
  })()`);
  await saveViewportScreenshot("Figure 4.3 - Analysis and Aspect Sentiment.png");

  await enterReview(
    "BEST PRODUCT EVER!!! Absolutely perfect!!! Amazing amazing amazing!!! Everyone should buy this immediately!!!",
    5,
    "N",
  );
  await submitReview();
  const fakeResult = await evaluate(`document.querySelector('[aria-labelledby="results-heading"]').innerText`);
  await saveViewportScreenshot("Figure 4.4 - Fake Deceptive Prediction.png");
  await savePipelineScreenshot("Figure 4.5 - Analysis Pipeline.png");

  console.log(JSON.stringify({ negativeResult, fakeResult }, null, 2));
  await send("Browser.close").catch(() => {});
  socket.close();
  chrome.kill();
}

main().catch((error) => {
  console.error(error);
  if (socket?.readyState === WebSocket.OPEN) socket.close();
  chrome.kill();
  process.exitCode = 1;
});
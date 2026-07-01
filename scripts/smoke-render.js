const { chromium } = require("playwright");
const fs = require("fs");

const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ||
  "/home/vmadmin/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome";
const url = process.env.PLAYGROUND_URL || "http://127.0.0.1:4173/";

(async () => {
  const launchOptions = {
    headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage"]
  };
  let browser;
  try {
    browser = await chromium.launch(launchOptions);
  } catch (e) {
    if (!fs.existsSync(executablePath)) {
      throw e;
    }
    browser = await chromium.launch(Object.assign({}, launchOptions, {
      executablePath
    }));
  }
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on("pageerror", err => errors.push("pageerror: " + err.message));
  page.on("console", msg => {
    if (msg.type() === "error") {
      errors.push("console: " + msg.text());
    }
  });

  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForSelector("#random-walk-projection canvas", {
    timeout: 15000
  });
  await page.waitForSelector("#output-3d-projection canvas", {
    timeout: 15000
  });
  await page.waitForSelector("#sweep-heatmap", { timeout: 15000 });
  await page.waitForFunction(() =>
    (document.querySelector("#loss-train") || {}).textContent.trim().length > 0,
    null,
    { timeout: 15000 });

  const result = await page.evaluate(() => {
    const projection = document.querySelector("#random-walk-projection canvas");
    const output3d = document.querySelector("#output-3d-projection canvas");
    const heatmap2d = document.querySelector("#heatmap .heatmap-2d");
    const sweep = document.querySelector("#sweep-heatmap");
    const projectionDataUrl = projection ? projection.toDataURL("image/png") :
      "";
    const output3dDataUrl = output3d ? output3d.toDataURL("image/png") :
      "";
    return {
      projectionCanvas: !!projection,
      projectionWidth: projection && projection.width,
      projectionHeight: projection && projection.height,
      projectionDataUrlLength: projectionDataUrl.length,
      output3dCanvas: !!output3d,
      output3dWidth: output3d && output3d.width,
      output3dHeight: output3d && output3d.height,
      output3dDataUrlLength: output3dDataUrl.length,
      output3dMode: (document.querySelector("#output-3d-mode") || {})
        .textContent || "",
      output3dLegend: (document.querySelector("#output-3d-legend") || {})
        .textContent || "",
      heatmap2dDisplay: heatmap2d ? getComputedStyle(heatmap2d).display : "",
      sweepCanvas: !!sweep,
      cnnToggle: !!document.querySelector("#useCnnFrontend"),
      transformerToggle: !!document.querySelector("#useTransformer"),
      randomWalkDescription: (document.querySelector(
        "#random-walk-description-title") || {}).textContent || "",
      lossTrainText: (document.querySelector("#loss-train") || {})
        .textContent || "",
      lossTestText: (document.querySelector("#loss-test") || {})
        .textContent || "",
      inputNodes: document.querySelectorAll("#svg g.node").length
    };
  });

  await page.hover("#network div.canvas");
  await page.waitForFunction(() =>
    /node color:/.test((document.querySelector("#output-3d-mode") || {})
      .textContent || ""),
    null,
    { timeout: 15000 });
  const output3dHoverMode = await page.$eval("#output-3d-mode",
    el => el.textContent.trim());

  await page.check("#useCnnFrontend");
  await page.waitForFunction(() =>
    document.querySelectorAll("#svg g.node").length >= 8,
    null,
    { timeout: 15000 });
  const cnnChecked = await page.isChecked("#useCnnFrontend");

  await page.check("#useTransformer");
  await page.waitForFunction(() =>
    document.querySelectorAll("#svg g.node").length >= 16,
    null,
    { timeout: 15000 });
  const toggleState = {
    cnnCheckedBeforeTransformer: cnnChecked,
    cnnAfterTransformer: await page.isChecked("#useCnnFrontend"),
    transformerChecked: await page.isChecked("#useTransformer")
  };

  await browser.close();

  const externalErrors = errors.filter(e =>
    !/ERR_CERT|ERR_CONNECTION_CLOSED|google-analytics|fonts\.googleapis|ERR_BLOCKED_BY_CLIENT|404 \(Not Found\)/.test(e));
  const failed = externalErrors.length ||
    !result.projectionCanvas ||
    !result.output3dCanvas ||
    result.output3dWidth < 290 ||
    result.output3dHeight < 290 ||
    result.output3dDataUrlLength < 1000 ||
    result.heatmap2dDisplay !== "none" ||
    !/Truth line: random walk/.test(result.output3dMode) ||
    !/model line color: MSE/.test(result.output3dMode) ||
    !/true walk line/.test(result.output3dLegend) ||
    !/node color:/.test(output3dHoverMode) ||
    !result.sweepCanvas ||
    !/N-Dimensional Random-Walk Regression/.test(
      result.randomWalkDescription) ||
    !toggleState.cnnCheckedBeforeTransformer ||
    toggleState.cnnAfterTransformer ||
    !toggleState.transformerChecked ||
    !result.lossTrainText ||
    !result.lossTestText;
  const report = { result, output3dHoverMode, toggleState, errors,
    externalErrors };
  if (failed) {
    console.error(JSON.stringify(report, null, 2));
    process.exit(1);
  }
  console.log(JSON.stringify(report, null, 2));
})();

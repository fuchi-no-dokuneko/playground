const { Builder, By } = require("selenium-webdriver");
const firefox = require("selenium-webdriver/firefox");
const geckodriver = require("geckodriver");
const fs = require("fs");
const net = require("net");

const firefoxBinary = process.env.FIREFOX_BINARY ||
  "/home/vmadmin/.cache/ms-playwright/firefox-1532/firefox/firefox";
const url = process.env.PLAYGROUND_URL || "http://127.0.0.1:4173/";
const port = Number(process.env.GECKODRIVER_PORT || 4444);

function waitForPort(port, host = "127.0.0.1", timeoutMs = 15000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    function tryConnect() {
      const socket = net.connect(port, host);
      socket.once("connect", () => {
        socket.end();
        resolve();
      });
      socket.once("error", () => {
        socket.destroy();
        if (Date.now() - start > timeoutMs) {
          reject(new Error(`Timed out waiting for ${host}:${port}`));
        } else {
          setTimeout(tryConnect, 100);
        }
      });
    }
    tryConnect();
  });
}

(async () => {
  if (!fs.existsSync(firefoxBinary)) {
    throw new Error(`Firefox binary not found: ${firefoxBinary}`);
  }

  const gecko = await geckodriver.start({
    port,
    host: "127.0.0.1",
    binary: firefoxBinary,
    log: "error"
  });
  let driver;
  try {
    await waitForPort(port);
    const options = new firefox.Options()
      .setBinary(firefoxBinary)
      .addArguments("-headless");
    driver = await new Builder()
      .forBrowser("firefox")
      .usingServer(`http://127.0.0.1:${port}`)
      .setFirefoxOptions(options)
      .build();

    await driver.get(url);
    await waitForAppReady(driver, "initial load");

    const controls = await driver.executeScript(() => {
      const ids = [
        "dimensionN", "walkLengthK", "sampleMultiplierM", "seed",
        "noiseEnabled", "noiseMean", "noiseVariance", "projectionX",
        "projectionY", "projectionZ", "projectionRotateX",
        "projectionRotateY", "projectionRotateZ", "projectionScale",
        "projectionRange", "deadNeuronEps", "sweepEpochs", "useCnnFrontend",
        "useTransformer"
      ];
      return ids.reduce((result, id) => {
        result[id] = !!document.getElementById(id);
        return result;
      }, {});
    });
    assert(Object.keys(controls).every(id => controls[id]),
      "Missing one or more required plan controls: " +
      JSON.stringify(controls));

    await setInputValue(driver, "dimensionN", 4, "input");
    await waitForOptions(driver, "#projectionX option", 4);
    await setInputValue(driver, "walkLengthK", 8, "input");
    await setInputValue(driver, "sampleMultiplierM", 2, "input");
    await setInputValue(driver, "seed", "firefox-plan-seed", "change");
    await setChecked(driver, "noiseEnabled", true);
    await setInputValue(driver, "noiseMean", 0.01, "change");
    await setInputValue(driver, "noiseVariance", 0.002, "change");
    await waitForAppReady(driver, "after random walk control updates");

    await setInputValue(driver, "projectionX", 0, "change");
    await setInputValue(driver, "projectionY", 1, "change");
    await setInputValue(driver, "projectionZ", 2, "change");
    await setInputValue(driver, "projectionRotateX", 35, "input");
    await setInputValue(driver, "projectionRotateY", -20, "input");
    await setInputValue(driver, "projectionRange", 1, "input");
    await setInputValue(driver, "projectionScale", 1.25, "input");

    const beforeStep = await driver.findElement(By.css("#iter-number"))
      .getText();
    await driver.findElement(By.css("#next-step-button")).click();
    await driver.wait(async () => {
      const value = await driver.findElement(By.css("#iter-number")).getText();
      return value !== beforeStep;
    }, 30000, "next-step button did not advance the iteration");

    await setInputValue(driver, "activations", "relu", "change");
    await setInputValue(driver, "deadNeuronEps", 0.000001, "change");
    await driver.findElement(By.css("#next-step-button")).click();
    await driver.wait(async () => {
      const text = await driver.findElement(By.css("#dead-neuron-stats"))
        .getText();
      return /L1:/.test(text);
    }, 30000, "dead-neuron monitor did not render layer stats");

    await setChecked(driver, "useCnnFrontend", true);
    await driver.wait(async () => {
      return await nodeCount(driver) >= 8;
    }, 30000, "CNN frontend did not expand the rendered network");
    const cnnChecked = await isChecked(driver, "useCnnFrontend");
    await setChecked(driver, "useTransformer", true);
    await driver.wait(async () => {
      return await nodeCount(driver) >= 16;
    }, 30000, "Transformer frontend did not expand the rendered network");
    const architectureState = {
      cnnCheckedBeforeTransformer: cnnChecked,
      cnnAfterTransformer: await isChecked(driver, "useCnnFrontend"),
      transformerChecked: await isChecked(driver, "useTransformer")
    };
    assert(architectureState.cnnCheckedBeforeTransformer &&
      !architectureState.cnnAfterTransformer &&
      architectureState.transformerChecked,
      "CNN/Transformer toggle exclusivity failed: " +
      JSON.stringify(architectureState));

    await setChecked(driver, "useTransformer", false);
    await setInputValue(driver, "sweepEpochs", 1, "change");
    await setInputValue(driver, "sampleMultiplierM", 1, "input");
    await driver.findElement(By.css("#sweep-depth-button")).click();
    await driver.wait(async () => {
      const rows = await driver.findElements(By.css("#sweep-results tbody tr"));
      const status = await driver.findElement(By.css("#sweep-status"))
        .getText();
      return rows.length === 16 && /complete/i.test(status);
    }, 30000);

    const result = await driver.executeScript(() => {
      const projection = document.querySelector("#random-walk-projection canvas");
      const output3d = document.querySelector("#output-3d-projection canvas");
      const heatmap2d = document.querySelector("#heatmap .heatmap-2d");
      const sweep = document.querySelector("#sweep-heatmap");
      const projectionDataUrl = projection ? projection.toDataURL("image/png") :
        "";
      const output3dDataUrl = output3d ? output3d.toDataURL("image/png") :
        "";
      const sweepDataUrl = sweep ? sweep.toDataURL("image/png") : "";
      return {
        title: document.title,
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
        sweepDataUrlLength: sweepDataUrl.length,
        sweepRows: document.querySelectorAll("#sweep-results tbody tr").length,
        sweepStatus: (document.querySelector("#sweep-status") || {})
          .textContent || "",
        cnnToggle: !!document.querySelector("#useCnnFrontend"),
        transformerToggle: !!document.querySelector("#useTransformer"),
        dimensionN: document.querySelector("#dimensionN").value,
        walkLengthK: document.querySelector("#walkLengthK").value,
        sampleMultiplierM: document.querySelector("#sampleMultiplierM").value,
        projectionRange: document.querySelector("#projectionRange").value,
        hashHasSeed: location.hash.indexOf("seed=firefox-plan-seed") !== -1,
        deadNeuronStats: (document.querySelector("#dead-neuron-stats") || {})
          .textContent || "",
        lossTrainText: (document.querySelector("#loss-train") || {})
          .textContent || "",
        lossTestText: (document.querySelector("#loss-test") || {})
          .textContent || ""
      };
    });

    if (!result.projectionCanvas || !result.sweepCanvas ||
        !result.output3dCanvas ||
        !result.cnnToggle || !result.transformerToggle ||
        !result.lossTrainText || !result.lossTestText ||
        result.dimensionN !== "4" || result.walkLengthK !== "8" ||
        result.sampleMultiplierM !== "1" || result.projectionRange !== "1" ||
        result.projectionWidth < 600 || result.projectionHeight < 350 ||
        result.output3dWidth < 290 || result.output3dHeight < 290 ||
        result.output3dDataUrlLength < 1000 ||
        result.heatmap2dDisplay !== "none" ||
        !/Truth line: random walk/.test(result.output3dMode) ||
        !/model line color: MSE/.test(result.output3dMode) ||
        !/true walk line/.test(result.output3dLegend) ||
        !result.hashHasSeed ||
        !/L1:/.test(result.deadNeuronStats) ||
        result.sweepRows !== 16 || !/complete/i.test(result.sweepStatus)) {
      console.error(JSON.stringify(result, null, 2));
      process.exit(1);
    }

    await driver.get(withAuditHash(url, "one-neuron", "#problem=regression&regressionDatasetKind=1" +
      "&dimensionN=3&walkLengthK=20&sampleMultiplierM=1&networkShape=1" +
      "&seed=one-neuron-loss-audit"));
    await waitForAppReady(driver, "one-neuron loss audit");
    const auditState = await driver.executeScript(() => {
      return {
        dimensionN: document.querySelector("#dimensionN").value,
        walkLengthK: document.querySelector("#walkLengthK").value,
        sampleMultiplierM: document.querySelector("#sampleMultiplierM").value,
        hash: location.hash
      };
    });
    assert(auditState.dimensionN === "3" && auditState.walkLengthK === "20" &&
      auditState.sampleMultiplierM === "1" &&
      auditState.hash.indexOf("networkShape=1") !== -1,
      "One-neuron audit did not load the requested state: " +
      JSON.stringify(auditState));
    const beforeOneNeuronStep = await driver.findElement(By.css("#iter-number"))
      .getText();
    await driver.findElement(By.css("#next-step-button")).click();
    await driver.wait(async () => {
      const value = await driver.findElement(By.css("#iter-number")).getText();
      return value !== beforeOneNeuronStep;
    }, 30000, "one-neuron audit did not advance to iteration 1");
    const oneNeuronLoss = await driver.findElement(By.css("#loss-train"))
      .getText();
    assert(!/^0(?:\.0+)?$/.test(oneNeuronLoss.trim()),
      "One-neuron epoch-1 train loss is displayed as zero: " + oneNeuronLoss);

    console.log(JSON.stringify({result, oneNeuronLoss}, null, 2));
  } finally {
    if (driver) {
      await driver.quit();
    }
    gecko.kill();
  }
})();

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function withAuditHash(baseUrl, auditName, hash) {
  const withoutHash = baseUrl.split("#")[0];
  const separator = withoutHash.indexOf("?") === -1 ? "?" : "&";
  return `${withoutHash}${separator}seleniumAudit=${encodeURIComponent(auditName)}${hash}`;
}

async function waitForAppReady(driver, label) {
  await driver.wait(async () => {
    return await driver.executeScript(() => {
      return !!document.querySelector("#random-walk-projection canvas") &&
        !!document.querySelector("#output-3d-projection canvas") &&
        !!document.querySelector("#loss-train");
    });
  }, 30000, `${label}: projection canvas and loss element did not mount`);
  await driver.wait(async () => {
    return await driver.executeScript(() => {
      const loss = document.querySelector("#loss-train");
      return !!loss && loss.textContent.trim().length > 0;
    });
  }, 30000, `${label}: training loss did not render`);
}

async function setInputValue(driver, id, value, eventName) {
  await driver.executeScript(`
    const el = document.getElementById(arguments[0]);
    el.value = String(arguments[1]);
    el.dispatchEvent(new Event(arguments[2], { bubbles: true }));
  `, id, value, eventName);
}

async function setChecked(driver, id, checked) {
  await driver.executeScript(`
    const el = document.getElementById(arguments[0]);
    if (el.checked !== arguments[1]) {
      el.click();
    }
  `, id, checked);
}

async function isChecked(driver, id) {
  return await driver.executeScript(`
    return document.getElementById(arguments[0]).checked;
  `, id);
}

async function nodeCount(driver) {
  const nodes = await driver.findElements(By.css("#svg g.node"));
  return nodes.length;
}

async function waitForOptions(driver, selector, expectedCount) {
  await driver.wait(async () => {
    const options = await driver.findElements(By.css(selector));
    return options.length === expectedCount;
  }, 30000, `Expected ${expectedCount} options for ${selector}`);
}

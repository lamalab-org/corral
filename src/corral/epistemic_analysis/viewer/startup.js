"use strict";
// Keep this independent of the main script so even a startup error is visible.
(function () {
  var banner = document.getElementById("startup");
  var message = document.getElementById("startup-message");
  var started = false;
  var timer = setTimeout(function () {
    fail("The viewer has not finished loading. Check that your SSH tunnel is still connected, then reload. If this persists, share the error shown here.");
  }, 35000);
  function fail(detail) {
    clearTimeout(timer);
    message.textContent = detail;
    banner.hidden = false;
  }
  window.corralStartup = {
    ready: function () {
      started = true;
      clearTimeout(timer);
      banner.hidden = true;
    },
    fail: fail
  };
  window.addEventListener("error", function (event) {
    if (event.target && event.target.tagName === "SCRIPT") {
      fail("A viewer script could not load. Reload this page. If it still fails, check browser blocking extensions and your SSH tunnel.");
    } else if (!started && event.message) {
      fail("Viewer startup failed: " + event.message + ". Reload this page; if this persists, share this error and your browser version.");
    }
  }, true);
  window.addEventListener("unhandledrejection", function (event) {
    if (!started) fail("Viewer startup failed: " + (event.reason && event.reason.message || String(event.reason)));
  });
}());

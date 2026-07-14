import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions

REPO_ROOT = Path(__file__).resolve().parents[1]
PORT = 8097
URL = f"http://127.0.0.1:{PORT}/results/dashboard.html"


def _build_driver():
    browser_attempts = [
        ("chrome", webdriver.Chrome, ChromeOptions, ["--headless=new"]),
        ("edge", webdriver.Edge, EdgeOptions, ["--headless=new"]),
        ("firefox", webdriver.Firefox, FirefoxOptions, ["--headless"]),
    ]
    errors: list[str] = []
    for name, constructor, options_cls, args in browser_attempts:
        try:
            options = options_cls()
            for arg in args:
                options.add_argument(arg)
            return constructor(options=options)
        except Exception as exc:  # pragma: no cover
            errors.append(f"{name}: {exc}")
    pytest.skip("No local Selenium browser could be launched: " + " | ".join(errors))


@pytest.fixture(scope="module")
def static_server():
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1"],
        cwd=str(REPO_ROOT),
    )
    time.sleep(2)
    try:
        yield
    finally:
        server_proc.terminate()
        server_proc.wait(timeout=10)


def test_dashboard_ui_integrity(static_server):
    driver = _build_driver()
    try:
        driver.get(URL)
        time.sleep(2)

        # Check title
        assert "Maternal Anemia" in driver.title

        # Check NHANES active by default
        calc_title = driver.find_element(By.ID, "calculator-title").text
        assert "US Cohort" in calc_title

        # Validate NHANES default values / cohort size
        cohort_size_text = driver.find_element(By.ID, "metric-n").text
        assert cohort_size_text == "335"

        # Click Uganda toggle button
        uganda_btn = driver.find_element(By.ID, "btn-uganda")
        uganda_btn.click()
        time.sleep(1)

        # Verify cohort switched to Uganda
        calc_title_updated = driver.find_element(By.ID, "calculator-title").text
        assert "Uganda Cohort" in calc_title_updated

        # Verify Uganda cohort metrics updated
        cohort_size_updated = driver.find_element(By.ID, "metric-n").text
        assert cohort_size_updated == "1,000"

        # Check that Uganda-specific fields are now visible
        water_group = driver.find_element(By.ID, "water-group")
        assert water_group.is_displayed()

    finally:
        driver.quit()

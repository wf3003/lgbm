module.exports = {
  apps: [
    {
      name: "lgbm-strategy",
      script: "strategy.py",
      interpreter: "/home/rose/lgbm/venv/bin/python3",
      cwd: "/home/rose/lgbm",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      autorestart: true, max_restarts: 10, restart_delay: 5000,
      watch: false, time: true,
    },
    {
      name: "lgbm-dashboard",
      script: "dashboard.py",
      interpreter: "/home/rose/lgbm/venv/bin/python3",
      cwd: "/home/rose/lgbm",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },
  ],
};

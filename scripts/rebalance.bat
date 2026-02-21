@echo off
cd /d "C:\ai quantagent"
echo [%date% %time%] Rebalance started >> logs\rebalance.log
python main.py execute --confirm >> logs\rebalance.log 2>&1
echo [%date% %time%] Rebalance finished >> logs\rebalance.log
echo. >> logs\rebalance.log

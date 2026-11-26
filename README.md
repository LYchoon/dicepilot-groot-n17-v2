# DicePilot GR00T N1.7 v2

SO-101 骰子操作推論範例。模型使用側面及手腕相機、六個關節讀數和文字指令。

公開權重與雜湊位於 [Hugging Face](https://huggingface.co/LY0613/dicepilot-groot-n17-v2)。GitHub 提供載入程式與使用說明。權重是已訓練參數，需搭配 NVIDIA 基礎模型。

## 安裝與下載

在 Linux CUDA 環境依 [官方 N1.7 安裝說明](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.7-release) 安裝 GR00T。在該 Python 環境執行本範例。先取得 NVIDIA GR00T-N1.7-3B 及 Cosmos-Reason2-2B 的 Hugging Face 存取權，執行 hf auth login 登入。

```bash
git clone https://github.com/LYchoon/dicepilot-groot-n17-v2.git
cd dicepilot-groot-n17-v2
python infer.py --variant bc \
  --side side.jpg --wrist wrist.jpg \
  --state 0 0 0 0 0 50 \
  --task "Move the red die."
```

首次執行自動下載固定版本的基礎權重、BC 權重和 processor。上面的關節數值與文字是介面範例，使用時填入同一時間取得的實際關節讀數及任務。

可用 --base-dir 和 --weights-dir 指定已下載目錄。基礎模型 revision 固定為 373d34e2723d928870157d07dc216a19a624b5bb。

## 選擇權重

| --variant | 載入順序 | 匯出時的驗證結果 |
|---|---|---|
| bc | NVIDIA base → BC step 15000 | 12/50，24%，BC best 與 last 相同 |
| nft-best | NVIDIA base → BC → NFT action head | RL 當時最高分 10/50，20%，最佳 epoch 8 |
| nft-last | NVIDIA base → BC → NFT action head | 匯出 epoch 87，該 epoch 未獨立驗證 |

預設使用 bc。NFT 的兩份權重涵蓋整個 action head。快照的 last 固定為 epoch 87。模型庫的 checkpoints.json 記錄對應版本。這些結果來自模擬器的完整任務成功條件。

## 輸入與輸出

- side 和 wrist 為同步 RGB 相片，模擬器使用 640×480。
- state 順序為 shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper。
- 五個手臂關節使用 LeRobot SO-101 正規化讀數，範圍 -100 到 100。夾爪為 0 到 100。真機的校準需與資料收集時一致。
- processor 負責影像處理、狀態統計和相對動作解碼。輸出已解碼為 LeRobot 關節目標。
- 程式輸出 8×6 JSON 陣列，每列是一個動作。使用四步 flow integration，每個 chunk 執行前八步後重新觀測。

## 整合控制迴圈

```python
from infer import load_policy, predict

policy = load_policy("bc")
# side_rgb 和 wrist_rgb 為 H×W×3 uint8 numpy arrays。
# joint_state 為上述順序的六個 LeRobot 讀數。
actions = predict(policy, side_rgb, wrist_rgb, joint_state, instruction)
```

將八列目標依資料收集的控制週期送進既有機器人介面，再取得新影像與關節狀態。部署端需沿用既有 SO-101 校準及相機配置。

推論範例支援輸出動作。Jetson Thor 的實機控制迴圈與端到端成功率仍需在部署環境驗證。NVIDIA 基礎模型與相依模型適用各自的授權條款。

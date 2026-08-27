# 控制面安全与播放时 AEC

## 信任边界

| 边界 | 数据 | 默认保护 | 主要风险 |
| --- | --- | --- | --- |
| `neteasecli` → API Server `:9092` | 播放 URL、控制命令 | loopback + 256-bit Bearer token | 未授权控制、SSRF、超长输入 |
| 音箱 Client → 原生 WebSocket `:4399` | 16 kHz 麦克风、24 kHz 播放 PCM、RPC | 仅 loopback；远程需 WSS 反向代理 | 音频泄露、远程命令执行、token 窃听 |
| Bridge → 音频 CDN | 音乐下载 | HTTPS；拒绝私网/本机目标 | SSRF、内存耗尽、凭据头转发 |
| Bridge 进程内部 | 扬声器参考、麦克风近端音频 | 只驻留内存，普通日志不记录 PCM | 敏感音频泄露 |

API Server 首次启用时生成 256-bit token，默认保存在
`~/.config/open-xiaoai-bridge/api-token`，目录权限为 `0700`、文件权限为
`0600`。token 不会写入日志。轮换时停止 bridge，替换该文件为新的随机值并
保持 `0600`，再同时重启 bridge 与客户端。

所有 API（包括 health）都要求：

```text
Authorization: Bearer <token>
```

也可用 `OPENXIAOAI_API_TOKEN` 直接提供至少 32 字符的 token，或用
`OPENXIAOAI_API_TOKEN_FILE` 指定 secret 文件。推荐文件或容器 secret，避免把
token 放进 shell history、进程参数或镜像。

## 网络部署

- 默认 API 和原生 WebSocket 都只监听 `127.0.0.1`。
- API 设置为非 loopback 地址时，必须同时配置 `API_SERVER_TLS_CERT` 和
  `API_SERVER_TLS_KEY`；配置 `API_SERVER_CLIENT_CA` 后启用 mTLS。
- 原生 `:4399` 目前不在进程内终止 TLS。跨主机时保持
  `OPEN_XIAOAI_BIND=127.0.0.1:4399`，在同机用支持 WebSocket 的 TLS/mTLS
  反向代理暴露 WSS，并给音箱 Client 配置对应 CA 和 Bearer token。
- `OPEN_XIAOAI_ALLOW_INSECURE_WS=1` 与
  `OPENXIAOAI_ALLOW_INSECURE_HTTP=1` 只用于限时迁移；它们会让凭据和音频暴露
  给同网段监听者。

API 还会限制请求体大小，只转发短 `User-Agent`/`Referer` 头，拒绝内嵌凭据、
本机、私网、链路本地和云元数据目标。透明代理可能把公网媒体 CDN 解析到
`198.18.0.0/15` Fake-IP。此时优先使用
`STREAM_PLAYER_TRUSTED_HOSTS=music.126.net` 这类逗号分隔的显式域名/后缀白名单；
匹配项允许其子域名通过，但其他非公网地址仍会被拒绝。仅在整个目标网络均可信且
无法使用域名白名单时，才设置 `STREAM_PLAYER_ALLOW_PRIVATE_URLS=1` 并把 API 隔离
在可信网络中。

## AEC 音频链路

```text
媒体 URL → 解码 24 kHz/s16le/mono → AEC 远端参考 ─┐
                               └→ 音箱播放           │
音箱麦克风 16 kHz/s16le/mono → 频域 NLMS AEC ←──────┘
                              → 输入增益 → KWS/VAD/ASR
```

播放器每次送出的 60 ms PCM 块都会先写入参考队列。AEC 将其重采样到 16 kHz，
以 10 ms 块学习房间脉冲响应。`audio_input.aec.delay_ms` 补偿播放器缓冲、网络
和声学路径的固定延迟；双讲和采集削波时冻结自适应，防止把用户声音学进回声
模型。参考欠载/溢出计数、双讲、削波和学习块数可在认证后的 `/api/health`
查看。参考队列有 3 秒上限，暂停、seek、stop 时会清除滤波器和陈旧参考。

当前 AEC 只覆盖 `StreamPlayer` 的可观测 PCM（也就是 neteasecli 中转播放）。
音箱固件自行播放的媒体和 TTS 没有可靠 PCM 参考，不能声称已做 AEC；这些路径
仍使用现有的播放期关麦/播放结束清缓冲策略。长期时钟漂移会表现为持续的
underflow/overflow，需要校准 `delay_ms` 或在设备端提供带时间戳的播放回调。

## 实机校准与验收

先用认证后的 `/api/health` 确认 `aec.enabled=true`，再在同一房间、音量和位置
分别运行 AEC 关闭/开启两轮。每轮至少播放 5 分钟音乐，每 10 秒说一次唤醒词，
记录：设备型号/固件、房间、音量百分比、距离、成功次数/总次数、P50/P95 唤醒
延迟、CPU/RSS，以及 AEC diagnostics 前后差值。

建议验收表：

| 场景 | AEC off 成功率 | AEC on 成功率 | P95 延迟 | 备注 |
| --- | ---: | ---: | ---: | --- |
| 1 m / 40% 音量 | 待实测 | 待实测 | 待实测 | 标准场景 |
| 3 m / 60% 音量 | 待实测 | 待实测 | 待实测 | 远场 |
| 双讲 | 待实测 | 待实测 | 待实测 | 连续说话 |
| 扬声器削波 | 待实测 | 待实测 | 待实测 | 不应发散 |

自动化测试只证明格式、参考对齐、自适应收敛、双讲/削波保护和缓冲边界；真实
唤醒率取决于设备扬声器、麦克风、房间和音量，不能用合成测试替代。

开发环境微基准（KVM 中 4 vCPU AMD EPYC 7K62、Python 3.11、NumPy 2.4.6、
SciPy 1.17.1）处理 60 秒 16 kHz 音频耗时 1.135 秒，即单核实时占比约 1.89%，
每个 10 ms 块平均 0.189 ms，进程峰值 RSS 增量约 2.8 MiB。AEC 在现有采集
回调内同步处理，不额外排队完整音频帧；真实设备的 CPU、端到端唤醒延迟和播放
质量仍必须按上表复测。

### 任务
帮我完成router_service的后端、Dockerfile、yml文件脚本

### 功能描述
我希望router service 起到一个中继的功能，能够接收请求，先将请求经过yolov13_formula_detection_service，得到结果后将json中的数据划分为公式、表格和图片，之后将公式分发至pp-formulanet-plus-l 服务、表格分发至table_latex_service，最后合并结果，我希望数据的接收和分发是异步的，router service本身既能分发和分流请求，也能像缓冲池一样使得数据分发和请求分发形成流水线，即yolov13_formula_detection_service作为生产者生产数据，pp-formulanet-plus-l和table_latex_service作为消费者消费数据，最后router service 得到具体结果后返回总数据，我希望pp-formulanet-plus-l返回latex公式之后能够在原本的json中增加一项，同理table service 也是，最后呈现的结果是在 yolov13_formula_detection_service 的返回结果中增加pp-formulanet-plus-l返回的latex公式内容和table_latex_service的返回内容

#### 1. 接收请求和分流发送请求
router service 对接的服务有三个 yolov13_formula_detection_service、pp-formulanet-plus-l、slanet_plus_service，router service 接收到请求和数据（pdf的渲染图片）后，将数据和请求分发至 yolov13_formula_detection_service，得到返回结果后，对结果进行分类，公式一类，表格一类，保留原返回结果格式，提取公式框信息到公式缓存池，提取表格框信息到表格缓存池，之后公式缓存池中的公式框信息发送到 pp-formulanet-plus-l，表格框信息发送至slanet_plus_service，得到返回结果后，将结果重新合并到结果格式中，返回。

我希望这三个服务能够建立起流水线，以缩短数据处理时间

#### 2. 缓存池
公式缓存池：用于存储提取的公式框信息
表格缓存池：用于存储提取的表格框信息

#### 3. 多请求
我希望router service 能够同时接收 2个不同客户端的请求，能够使用session id区分不同客户端请求的pdf文件的公式框和表格框提取结果

## koa-starter

Koa项目脚手架模板，快速搭建基于Koa的中小型后端API接口



**默认中间件**：

koa-logger、@koa/cors、koa-static、koa-body、koa-jwt、koa-parameter、koa-router



**项目结构**：

```bash
|-- package-lock.json
|-- package.json
|-- readme.md
|-- src
    |-- app.js						# 项目入口
    |-- config						# 全局配置（jwt、数据库）
    |   |-- index.js
    |-- constant					# 全局常量
    |   |-- index.js
    |-- controllers					# 控制层/逻辑层
    |   |-- upload.controller.js	# 文件上传
    |-- db							# 数据库连接
    |   |-- index.js
    |-- middlewares					# 自定义中间件
    |   |-- error-handle.js			# API异常响应中间件
    |   |-- not-found.js			# 404响应中间件
    |   |-- result.js				# 常用响应函数
    |-- models						# mongoose数据库表
    |-- publ
    |   |-- images					# 图片上传存储
    |-- routes						# 路由（配置api url和controller层的映射关系）
    |   |-- example.js
    |   |-- index.js
    |   |-- upload.js
    |-- utils						# 全局通用函数/工具函数
        |-- auth.js

```



**运行**：

```shell
npm install
npm run dev
```



**端口**：3000

**example url**：http://localhost:3000/api/v1/example



## api根路径

`http://localhost:3000/api/v1/xxx`

在`routes`文件夹中编写的子路由URL会被`routes/index.js`的代码自动拼接到`http://localhost:3000/api/v1/`路径下。

```javascript
const Router = require('koa-router')
const { ExampleController } = require('../controllers/example')

const exampleRoute = new Router({
  prefix: '/example'    // 建议每个子路由设置一个url前缀
})

exampleRoute.get('/', ExampleController.example)	//http://localhost:3000/api/v1/example/
exampleRoute.get('/todos', ExampleController.todos)	//http://localhost:3000/api/v1/example/todos

module.exports = {
  exampleRoute
}
```



## routes和controller的关系

- controller：编写逻辑处理
- routes：导入controller，设置url



`controller/example.controller.js`

```javascript
class ExampleController {
  static async example(ctx, next) {
    // 参数校验...
    // ctx.verifyParams({
    //   name: { type: 'string', required: true },
    // })

    // 获取参数...
    // const param1 = ctx.request.body.param1 || ""

    // 业务逻辑/数据库操作...
    // throw new Error('test error')

    // 返回请求结果...
    ctx.result({
      code: 200,
      message: "exmple api"
    })
  }
}

module.exports = {
  ExampleController
}
```



`routes/example.js`

```javascript
const Router = require('koa-router')
const { ExampleController } = require('../controllers/example') // 导入controller

const exampleRoute = new Router({
  prefix: '/example'    // 建议每个子路由设置一个url前缀
})

exampleRoute.get('/', ExampleController.example)  //http://localhost:3000/api/v1/example/

module.exports = {
  exampleRoute
}
```



## 响应函数

`middlewares/result.js`中为`ctx`对象挂载了统一数据结构的响应函数。方便用于数据响应

- **succ({ data, message })**：请求处理成功，code:200
- **err({ data, message })**：请求处理失败，code:500
- **result({ code, data, message })**：通用请求响应



> **<font color='red'>注意</font>**：这些函数响应的http码都为200，处理状态应该通过响应数据的`code`属性设置



**响应数据结构**：

```javascript
{
    code,		// 处理状态码
    data,		// 响应数据
    message   	// 错误信息
}
```



**succ()使用示例**：

```javascript
class ExampleController {
  static async example(ctx, next) {
	// ...

    // 返回请求结果...
    ctx.succ({
        data:{
            
        },
        message:""	//message可省略
    })
  }
}

module.exports = {
  ExampleController
}
```



**err()使用示例**：

```javascript

class ExampleController {
  static async example(ctx, next) {
    // ...

    // 返回请求结果...
    ctx.err({
      message: ""
    })
  }
}

module.exports = {
  ExampleController
}
```



**result()使用示例**：

```javascript
class ExampleController {
  static async example(ctx, next) {
	// ...

    // 返回请求结果...
    ctx.result({
      code: 200,
      message: "exmple api"
    })
  }
}

module.exports = {
  ExampleController
}
```



## api版本号

`constant/index.js`中的`API_VERSION`变量设置了api版本号，可以修改url中的版本号

```javascript
module.exports = {
  API_VERSION: "v1"		//api版本号
}
```



## 参考/引用

https://www.bilibili.com/video/BV13A411w79h

[Koa中文网](https://www.koajs.com.cn/#)
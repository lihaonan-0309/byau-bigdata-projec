const Koa = require('koa')
const path = require('path')
const logger = require('koa-logger')      // 请求日志
const cors = require('@koa/cors')         // 允许跨域
const staticServe = require('koa-static')
const { koaBody } = require('koa-body')   // request body parsing
const koaJwt = require('koa-jwt')
const parameter = require('koa-parameter')
const { baseRoute } = require('./routes')
const { errorHandle } = require('./middlewares/error-handle')
const { notFound } = require('./middlewares/not-found')
const { result } = require('./middlewares/result')

// config
const config = require('./config')

// connect db
require('./db')

// app
const app = new Koa()

// use middleware
app.use(logger())
app.use(cors())
app.use(koaBody())
app.use(result)   // http response result methods
app.use(staticServe(path.join(__dirname, "/public")))
app.use(errorHandle)
// app.use(koaJwt({ secret: config.jwt.secret }).unless({ path: config.jwt.unless })) // 身份验证
parameter(app); // add verifyParams method, but don't add middleware to catch the error
// app.use(parameter(app)); // also add a middleware to catch the error.

// base route
app.use(baseRoute.routes()).use(baseRoute.allowedMethods())

// 404 middleware 放在route后面
app.use(notFound)

// start listen
app.listen(8081, () => {
  console.log('server is running on http://localhost:8081');
})
const Router = require('koa-router')
const { ExampleController } = require('../controllers/example') // 导入controller

const exampleRoute = new Router({
  prefix: '/example'    // 建议每个子路由设置一个url前缀
})

exampleRoute.get('/', ExampleController.example)  // http://localhost:3000/api/v1/example/

module.exports = {
  exampleRoute
}
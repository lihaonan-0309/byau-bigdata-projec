const Router = require('koa-router')
const { MapController } = require('../controllers/map') // 导入controller

const mapRoute = new Router({
    prefix: '/map'    // 建议每个子路由设置一个url前缀
})

mapRoute.get('/', MapController.map)  // http://localhost:3000/api/v1/map/

module.exports = {
    mapRoute
}
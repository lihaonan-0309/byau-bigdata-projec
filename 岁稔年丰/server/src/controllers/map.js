
const { mapModel } = require('../models/map');
class MapController {
  static async map(ctx, next) {
    const data = await mapModel.find();
    // 返回请求结果...
    ctx.result({
      code: 200,
      data,
    })
  }
}

module.exports = {
    MapController
}
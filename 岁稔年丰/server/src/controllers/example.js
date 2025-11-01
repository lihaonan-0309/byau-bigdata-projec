
const { exampleModel } = require('../models/exmple');
class ExampleController {
  static async example(ctx, next) {
    // 参数校验...
    // ctx.verifyParams({
    //   name: { type: 'string', required: true },
    //   age: { type: 'number', required: true },
    // })

    // 获取参数...
    // const param1 = ctx.request.body.param1 || ""
    // const param1 = ctx.query || ""
    const { typeId } = ctx.query;
    // 业务逻辑/数据库操作...
    // throw new Error('Example error!!!')
    // const createChart = await exampleModel.create({
    //   cid: 7,
    //   name:"测试",
    //   list: []
    // });
    const chart = await exampleModel.find({"cid": Number(typeId)});
    // 返回请求结果...
    ctx.result({
      code: 200,
      data: chart[0]
    })
  }
}

module.exports = {
  ExampleController
}
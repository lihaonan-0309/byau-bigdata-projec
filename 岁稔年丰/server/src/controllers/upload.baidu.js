const fs = require('fs')
const path = require('path')
const request = require('request')
const { axiosRequest } = require('../axios')

class UploadController {
  // 图片上传
  static async uploadImage(ctx, next) {
    // 获取key为image的图片
    const filesObj = ctx.request.files || {}
    let { file } = filesObj  // key = image
    if (!file) {
      ctx.err({ message: "no image param" })
      return
    }

    // 校验图片格式
    const fileTypes = ['image/jpeg', 'image/png']
    if (!fileTypes.includes(file.mimetype)) {
      ctx.err({ message: "the image format is not 'jpg', 'jpeg', 'png'" })
      return
    }

    // 查询哪个ai识别
    const { typeId } = ctx.query;

    // 读取上传的文件（文件会被koa-body上传至一个临时文件中）
    const reader = fs.createReadStream(file.filepath);

    // 获取图片上传后的目录路径
    const srcPath = __dirname.slice(0, __dirname.lastIndexOf("controllers"))
    // 获取图片后缀
    const suffix = file.originalFilename.substring(file.originalFilename.lastIndexOf('.'), file.originalFilename.length)
    const writePath = path.join(srcPath, '/public/images/', file.newFilename + suffix)

    // 保存图片
    const writeStream = fs.createWriteStream(writePath)
    reader.pipe(writeStream)

    // 图片的http链接
    const httpPath = ctx.origin + '/images/' + file.newFilename + suffix

    let ak = 'nzYtdPQGPNqrPFRyUur2GzaB';
    let sk = 'wDNRqb2RhOiKYOCf4VVdUowEIcnKosmz';

    let tokenRes = await axiosRequest(
      {
        method: 'POST',
        url: `https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=${ak}&client_secret=${sk}`
      }
    );
    let token = tokenRes.data.access_token;
    let config;

    switch (typeId) {
      case "1":
        config = {
          method: "POST",
          url: `https://aip.baidubce.com/rpc/2.0/ai_custom/v1/classification/bchjk?access_token=${token}&input_type=url`,
          data: JSON.stringify({
            url: httpPath,
            baike_num: 5
          }),
          headers: {
            'Content-Type': 'application/json'
          }
        }
        break;
      case "2":
        config = {
          method: "POST",
          url: `https://aip.baidubce.com/rest/2.0/image-classify/v1/plant?access_token=${token}`,
          data: {
            url: httpPath,
            baike_num: 5
          },
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
          }
        }
        break;
      default:
        break;
    }
    let res = await axiosRequest(config);

    ctx.succ({
      data: res.data
    })
  }
}

module.exports = {
  UploadController
}
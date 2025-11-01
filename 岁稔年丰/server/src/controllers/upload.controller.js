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

    // 读取上传的文件（文件会被koa-body上传至一个临时文件中）
    const reader = fs.createReadStream(file.filepath);

    // 获取图片上传后的目录路径
    const srcPath = __dirname.slice(0, __dirname.lastIndexOf("\\"))
    // 获取图片后缀
    const suffix = file.originalFilename.substring(file.originalFilename.lastIndexOf('.'), file.originalFilename.length)
    const writePath = path.join(srcPath, '/public/images/', file.newFilename + suffix)

    // 保存图片
    const writeStream = fs.createWriteStream(writePath)
    reader.pipe(writeStream)

    // 图片的http链接
    const httpPath = ctx.origin + '/images/' + file.newFilename + suffix

    let url;
    let res = await axiosRequest('POST', "http://127.0.0.1:5000/upload_image", { url: httpPath });
    url = res.data.image_url;
    let hurl = url.split('/');
    const name = hurl[hurl.length - 1];
    const writePath2 = `${srcPath}\\public\\aiImages\\${name}`;
    const writeStream2 = fs.createWriteStream(writePath2);
    request(url).pipe(writeStream2);

    // // 生成已保存文件的路径
    const aiHttpPath = ctx.origin + '/aiImages/' + name;

    ctx.succ({
      data: {
        path: aiHttpPath
      }
    })
  }
}

module.exports = {
  UploadController
}
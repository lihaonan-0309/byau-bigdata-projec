/*
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2023-08-02 14:58:23
 */
import axios from 'axios';
import UtilVar from "@/config/UtilVar";
import router from '@/router'

let { baseUrl, tianqiUrl } = UtilVar;
const CancelToken = axios.CancelToken;
export { baseUrl };
// axios.defaults.withCredentials = true;
// 添加请求拦截器
axios.interceptors.request.use(function(config) {
	config.headers['Content-Type'] = "application/json;charset=utf-8";
	return config;
}, function(error) {
	// 对请求错误做些什么
	console.log(error)
	return Promise.reject(error);
});
/**
 * @响应拦截
 */
axios.interceptors.response.use(response => {

	if (response.status !== 200) {
		return Promise.reject(response)
	}
	return response.data
}, error => {
	console.error(error);
	let err = {
		success: false,
		msg: "未知异常，请联系管理员！"
	}
	return Promise.reject(err)
})

let configs_ENC = {
	headers: { 'enc': UtilVar.ENC }
}
//处理是否加密数据
let isEncryptionParam = (params) => {
	return params

}
export const GET = async (url, params) => {
	try {
		const data = await axios.get(`${baseUrl}/${url}`, {
			headers: {
				"Content-Type": "application/json;charset=UTF-8"
			}
		});
		return data;
	} catch (error) {
		return error;
	}
}
export const TQGET = async (url, params) => {
	try {
		params = isEncryptionParam(params)
		const data = await axios.get(`${tianqiUrl}/${url}`, {
			params: params,
			headers: {
				"Content-Type": "application/json;charset=UTF-8"
			}
		});
		return data;
	} catch (error) {
		return error;
	}
}

export const GETNOBASEOTHER = async (url, params) => {
	try {
		const data = await axios.get(url, {
			params: params,
		});
		return data;
	} catch (error) {
		return error;
	}
}

//没有基地址 访问根目录下文件

export const GETNOBASE = async (url, params) => {
	try {
		const data = await axios.get(url, {
			params: params,
		});
		return data;
	} catch (error) {
		return error;
	}
}
export const POST = async (url, params) => {
	try {
		params = isEncryptionParam(params)
		const data = await axios.post(`${baseUrl}${url}`, params, configs_ENC);
		return data;
	} catch (error) {
		return error;
	}
}
export const PUT = async (url, params) => {
	try {
		params = isEncryptionParam(params)
		const data = await axios.put(`${baseUrl}${url}`, params, configs_ENC);
		return data;
	} catch (error) {
		return error;
	}
}
export const DELETE = async (url, params) => {
	// console.log(params)
	try {
		params = isEncryptionParam(params)
		const data = await axios.delete(`${baseUrl}${url}`, { data: params, headers: configs_ENC.headers },
			configs_ENC);
		return data;
	} catch (error) {
		return error;
	}
}


/**
 * @文件类型提交方法
 */
let configs = {
	headers: { 'Content-Type': 'multipart/form-data' },

}
export const FILESubmit = async (url, params, config) => {
	try {
		const data = await axios.post(`${baseUrl}${url}`, params, {
			...configs,
			cancelToken: new CancelToken(function executor(c) {
				config.setCancel && config.setCancel(c)
			}),
			onUploadProgress: (e) => {
				if (e.total > 0) {
					e.percent = e.loaded / e.total * 100;
				}
				// console.log(config)
				config.onProgress && config.onProgress(e)
			},

		});
		return data;
	} catch (err) {
		return err;
	}
}

/**
 * 下载文档流
 * @param {config.responseType} 下载文件流根据后端 配置   arraybuffer || blod
 */
export const FILE = async (config = {}, body, params) => {
	try {
		const data = await axios({
			method: config.method || 'get',
			url: `${baseUrl}${config.url}`,
			data: body,
			params: params,
			responseType: config.responseType || 'blob',
			onDownloadProgress: (e) => {
				// console.log(e,e.currentTarget)
				// if (e.currentTarget.response.size > 0) {
				//     e.percent = e.loaded / e.currentTarget.response.size * 100;
				// }
				// event.srcElement.getResponseHeader('content-length')
				config.onProgress && config.onProgress(e)
			},
		});
		return data;
	} catch (err) {
		return err;
	}
}
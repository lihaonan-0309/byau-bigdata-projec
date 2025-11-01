import axios, { type AxiosProxyConfig, type AxiosRequestConfig } from 'axios'

// const baseURL = "http://127.0.0.1:3001/api/v1";
const baseURL = import.meta.env.VITE_BASE_URL + "/api/v1";


export function request(config:AxiosRequestConfig):Promise<any> {
  const instance = axios.create({
    baseURL: baseURL,
    timeout: 150000,
    headers: {
      // 'Content-Security-Policy': 'upgrade-insecure-requests'
      // 'Content-Type': 'application/json'
      // Accept: "application/json, text/plain, *"
    }
  })
  
  //请求拦截
  instance.interceptors.request.use(config => {
    return config
  }, err => {
 
  })

  //响应拦截
  instance.interceptors.response.use(res => {
    return res.data
  }, err => {
  })


  //发送网络请求
  return instance(config)
}


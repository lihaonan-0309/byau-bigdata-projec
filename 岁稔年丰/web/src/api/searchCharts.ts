import { request } from '../request/index'

interface searchCharts {
    typeId: Number
}

export function searchCharts(p: searchCharts) {
    return request({
        method: 'get',
        url: '/example',
        params: p,
    })
}

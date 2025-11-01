const axios = require('axios')
async function axiosRequest(config) {
    // const req = axios({
    //     method,
    //     url,
    //     data,
    //     headers: {
    //         'Content-Type': 'application/x-www-form-urlencoded',
    //         'Accept': 'application/json'
    //     },
    // })
    const req = axios(config);
    return req;
}

module.exports = {
    axiosRequest
}
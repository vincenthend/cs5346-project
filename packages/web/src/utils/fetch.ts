import ky from 'ky'

const BASE_URL = 'http://localhost:5000'
export const appFetch = <T = any>(resource: string, init: any) => {
  return ky<T>(BASE_URL + resource, init).json()
}

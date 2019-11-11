import * as expServices from '../services/prediction'

const initialState = {
  result: "initial message"
}

export default {
  namespace: 'prediction',

  state: initialState,

  effects: {
    * show(action, { call, put }) {
      const { data } = yield call(expServices.show)
      console.log(data)
      if (data) {
        yield put({
          type: 'updateUserData',
          payload: {
            ...data,
          },
        })
      } else {
        yield put({ type: 'showFail' })
      }
    },
  },

  reducers: {
    updateUserData: (state, { payload }) => ({
      ...state,
      ...payload,
    }),
  },
}

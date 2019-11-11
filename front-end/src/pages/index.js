import { connect } from 'dva';
import { Form, Radio, InputNumber, Button } from 'antd'
import styles from './index.css';

const FormItem = Form.Item

const formItemLayout = {
  labelCol: {
    xs: { span: 24 },
    sm: { span: 6 },
  },
  wrapperCol: {
    xs: { span: 24 },
    sm: { span: 12 },
  },
}

const tailFormItemLayout = {
  wrapperCol: {
    xs: {
      offset: 24,
    },
    sm: {
      offset: 6,
    },
  },
}

const handleSubmit = (e) => {
  e.preventDefault()
  this.props.form.validateFieldsAndScroll((err, values) => {
    if (!err) {
      const { dispatch } = this.props
      dispatch({
        type: 'prediction/predict',
        payload: {
          ...values,
        },
      })
    }
  })
}

const Page = ({ dispatch, prediction, form }) => {
  const { getFieldDecorator } = form

  return (
    <div className={styles.normal}>
      <div>
        <Form {...formItemLayout} onSubmit={handleSubmit}>

          <FormItem label="Category">
            {getFieldDecorator('category', {
              rules: [{ type: 'integer', required: true, message: 'Please select an action', whitespace: true }],
              initialValue: 0,
            })(
              <Radio.Group disabled>
                <Radio value={0}>Movie</Radio>
                <Radio value={1}>Drama</Radio>
                <Radio value={2}>Other</Radio>
              </Radio.Group>
            )}
          </FormItem>
          <FormItem label="Budget">
            {getFieldDecorator('budget', {
              rules: [{ required: true, message: 'Please input your budget' }],
              initialValue: 2000000,
            })(
              <InputNumber />
            )}
            <span className="ant-form-text"> USD</span>
          </FormItem>
          <FormItem {...tailFormItemLayout}>
            <Button type="primary" htmlType="submit">
              Submit
            </Button>
          </FormItem>
        </Form>
      </div>
    </div>
  );
}

export default connect(({ prediction }) => ({ prediction }))(Form.create()(Page))

import { connect } from 'dva';
import { Form, Radio, InputNumber, Button } from 'antd';
import { Chart, Geom, Axis, Tooltip, Legend, Coord } from 'bizcharts';
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

const cols = {
  probability: { alias: 'Probability' },
  status: { alias: 'Status' }
};

const Page = ({ dispatch, prediction, form }) => {
  const { getFieldDecorator } = form

  const handleSubmit = (e) => {
    e.preventDefault()
    form.validateFieldsAndScroll((err, values) => {
      if (!err) {
        dispatch({
          type: 'prediction/predict',
          payload: {
            ...values,
          },
        })
      }
    })
  }

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
      <Chart width={600} height={400} data={prediction.boxOfficeData} scale={cols}>
        <Axis name="status" title/>
        <Axis name="probability" title/>
        <Legend position="top" dy={-20} />
        <Tooltip />
        <Geom type="interval" position="status*probability" color="status" />
      </Chart>
    </div>
  );
}

export default connect(({ prediction }) => ({ prediction }))(Form.create()(Page))

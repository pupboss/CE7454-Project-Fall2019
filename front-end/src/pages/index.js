import { connect } from 'dva';
import { Select, Input, Icon, message, Upload, Row, Col, Rate, Form, Radio, InputNumber, Button } from 'antd';
import { Chart, Geom, Axis, Tooltip, Legend } from 'bizcharts';
import styles from './index.css';

const FormItem = Form.Item
const { Option } = Select

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

const getUploadData = (file) => {
  console.log(file)
  return {
    smfile: file
  }
}

const uploadProps = {
  data: getUploadData,
  onChange(info) {
    if (info.file.status !== 'uploading') {
      console.log(info.file);
    }
    if (info.file.status === 'done') {
      message.success(`${info.file.name} file uploaded successfully`);
    } else if (info.file.status === 'error') {
      message.error(`${info.file.name} file upload failed.`);
    }
  },
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
          <FormItem label="Poster">
            <Upload {...uploadProps}>
              <Button>
                <Icon type="upload" /> Click to Upload
              </Button>
            </Upload>
          </FormItem>
          <FormItem label="Title">
            {getFieldDecorator('title', {
              rules: [{ required: true, message: 'Please input the title' }],
            })(
              <Input />
            )}
          </FormItem>
          <FormItem label="Year">
            {getFieldDecorator('year', {
              rules: [{ required: true, message: 'Please input the year' }],
              initialValue: 2019,
            })(
              <InputNumber />
            )}
          </FormItem>
          <FormItem label="Duration">
            {getFieldDecorator('duration', {
              rules: [{ required: true, message: 'Please input the duration' }],
              initialValue: 120,
            })(
              <InputNumber />
            )}
            <span className="ant-form-text"> mins</span>
          </FormItem>
          <FormItem label="Director">
            {getFieldDecorator('director', {
              rules: [{ required: true, message: 'Please select the director' }],
            })(
                <Select
                  showSearch
                  style={{ width: 200 }}
                  placeholder="Select a director"
                >
                  <Option value="jack">Jack</Option>
                  <Option value="lucy">Lucy</Option>
                  <Option value="tom">Tom</Option>
                </Select>
            )}
          </FormItem>
          <FormItem label="Budget">
            {getFieldDecorator('budget', {
              rules: [{ required: true, message: 'Please input your budget' }],
              initialValue: 2000000,
            })(
              <InputNumber step={100000} style={{ width: '140px' }} />
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
      <div style={{ paddingTop: '50px' }}>
        <Row>
          <Col span={12}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ paddingTop: '90px' }}>
                Predict: <Rate allowHalf disabled count={10} value={8.5} /> 8.5 Stars
              </div>
              <div>
                Actual: <Rate allowHalf disabled count={10} value={8} /> 8 Stars
              </div>
            </div>
          </Col>
          <Col span={12}>
            <div>
              <Chart width={600} height={400} data={prediction.boxOfficeData} scale={cols}>
                <Axis name="status" title/>
                <Axis name="probability" title/>
                <Legend position="top" dy={-20} />
                <Tooltip />
                <Geom type="interval" position="status*probability" color="status" />
              </Chart>
            </div>
          </Col>
        </Row>
      </div>
    </div>
  );
}

export default connect(({ prediction }) => ({ prediction }))(Form.create()(Page))

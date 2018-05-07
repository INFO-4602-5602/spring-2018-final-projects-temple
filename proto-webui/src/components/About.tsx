import * as React from 'react';
import { Button, Header, Icon, Image, Modal } from 'semantic-ui-react';

import logo from '../logo.svg';

export default class About extends React.Component {
	public render() {
		return (
			<Modal dimmer="blurring" trigger={<Button inverted={true}><Icon name="info circle" />About This</Button>}>
				<Modal.Header><Icon name="info circle" />About This Tool</Modal.Header>
				<Modal.Content image={true}>
					<Image wrapped={true} size="large" src={logo} />
					<Modal.Description>
						<Header>Bayesian Optimization Tutorial</Header>
						<p>This tool was designed to support learning. It is not meant to be used for analysis.
						  The implementation of this tool is heavily inspired by Nando de Freitas's implementation
						  of Bayesian optimization, which you can find
              			  <a href="http://www.cs.ubc.ca/~nando/540-2013/lectures/gp.py"> on his website</a>.</p>

						<p>This software was developed using React.js and GPU.js by William Temple at the University of Colorado Boulder
						  using course material from Prof. Michael Mozer. It was developed as coursework for Dr.
              			  Danielle Szafir.</p>
						
						<p style={{ float: "right" }}><img src="/agplv3-155x51.png" /></p>
					</Modal.Description>
				</Modal.Content>
			</Modal>
		)
	}
}
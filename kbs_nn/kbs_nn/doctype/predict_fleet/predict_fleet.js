// Copyright (c) 2016, Steve Nyaga and contributors
// For license information, please see license.txt
frappe.provide('knbs_nn.neural');
frappe.ui.form.on('Predict Fleet', {
	refresh: function(frm) {
		frm.disable_save();			
	},
	onload_post_render: function(frm){
		frm.fields_dict.predict.$input.addClass('btn-primary')	
	}
});

frappe.ui.form.on("Predict Fleet", "predict", function(frm) {
	knbs_nn.neural.check_mandatory_to_fetch(frm.doc);
    return frappe.call({
		method: 'do_predict',
		doc: frm.doc,
		freeze_message: __('Predicting...Please wait'),
		freeze: true,
		callback: function(r) {			
			var doclist = frappe.model.sync(r.message);
			frappe.model.set_value(frm.doc.doctype, frm.doc.name, 'predicted_value', r.message.predicted_value);
			frappe.model.set_value(frm.doc.doctype, frm.doc.name, 'accuracy', r.message.accuracy);

		    /*frm.fields_dict.process_bill.$input.addClass("btn-primary");
			var doclist = frappe.model.sync(r.message);
			frappe.set_route("Form", doclist[0].doctype, doclist[0].name);
			*/
		}
	});
});

knbs_nn.neural.check_mandatory_to_fetch = function(doc) {
	$.each(["inputs", "iterations", "learning_rate"], function(i, field) {
		if(!doc[frappe.model.scrub(field)]) frappe.throw(__("Please select {0} first", [field]));
	});
}

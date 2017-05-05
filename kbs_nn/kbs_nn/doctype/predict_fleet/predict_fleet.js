// Copyright (c) 2016, Steve Nyaga and contributors
// For license information, please see license.txt
frappe.provide('knbs_nn.neural');
frappe.ui.form.on('Predict Fleet', {
	refresh: function(frm) {
		frm.disable_save();					
	},
	onload_post_render: function(frm){
		frm.fields_dict.predict.$input.addClass('btn-primary')	
		frm.toggle_display(['download_log_file', false])
	}
});

frappe.ui.form.on("Predict Fleet", "predict", function(frm) {
	knbs_nn.neural.check_mandatory_to_fetch(frm.doc);
	frm.toggle_display(['download_log_file', false])
    return frappe.call({
		method: 'do_predict',
		doc: frm.doc,
		freeze_message: __('Predicting...Please wait'),
		freeze: true,
		callback: function(r) {			
			var doclist = frappe.model.sync(r.message);
			frappe.model.set_value(frm.doc.doctype, frm.doc.name, 'predicted_value', r.message.predicted_value);
			frappe.model.set_value(frm.doc.doctype, frm.doc.name, 'accuracy', r.message.accuracy);
			
			frm.toggle_display(['download_log_file', true])			
		}
	});
});

frappe.ui.form.on("Predict Fleet", "download_training_data", function(frm){	
	knbs_nn.neural.do_download("/private/files/fleet_modified.xlsx");
});

frappe.ui.form.on("Predict Fleet", "download_log_file", function(frm){

	knbs_nn.neural.do_download("/private/files/kbs_nn_log.txt")
});

knbs_nn.neural.do_download = function(filename){
	var a = document.createElement('a');
	a.href = filename// 'private/files/kbs_nn_log.txt'// 'data:attachment/csv,' + encodeURIComponent(csv_data);
	a.download = filename;
	a.target = "_blank";
	document.body.appendChild(a);
	a.click();

	document.body.removeChild(a);
};

knbs_nn.neural.check_mandatory_to_fetch = function(doc) {
	$.each(["inputs", "iterations", "learning_rate"], function(i, field) {
		if(!doc[frappe.model.scrub(field)]) frappe.throw(__("Please select {0} first", [field]));
	});
}

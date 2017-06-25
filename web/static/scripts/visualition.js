$(document).ready(function(){
    selected = $('#network_select').find(':selected').text();

    $("[name="+selected+"]").css('display','block');
    $('#network_select').on('change',function(){
        $('.layers').removeAttr('style');
        $("[name="+selected+"]").css('display','block');
    });
});